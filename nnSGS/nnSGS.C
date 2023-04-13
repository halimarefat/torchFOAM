/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "nnSGS.H"
#include "fvOptions.H"
#include "wallDist.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void nnSGS<BasicTurbulenceModel>::correctNut
(
    const tmp<volTensorField>& gradU
)
{
    this->nut_ = max(Cs_*sqr(this->delta())*mag(dev(symm(gradU))),-1.0*this->nu());
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void nnSGS<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
nnSGS<BasicTurbulenceModel>::nnSGS
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    Cs_
    (
        IOobject
        (
            IOobject::groupName("Cs", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar ("Cs", dimless,SMALL)
    ),
    simpleFilter_(U.mesh()),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_()),
    y_(wallDist::New(this->mesh_).y())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool nnSGS<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());        

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
void nnSGS<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    LESeddyViscosity<BasicTurbulenceModel>::correct(); // For changing mesh - coming from turbulenceModel.C through virtual shit

    int in_channels = 9;
    int out_channels = 1;

    tmp<volTensorField> tgradU(fvc::grad(this->U_));
    const volTensorField& gradU = tgradU();

    int cellNum = this->mesh_.cells().size();

    volSymmTensorField S(dev(symm(gradU)));
    volScalarField S11 = S.component(tensor::XX);
    volScalarField S12 = S.component(tensor::XY);
    volScalarField S13 = S.component(tensor::XZ);
    volScalarField S22 = S.component(tensor::YY);
    volScalarField S23 = S.component(tensor::YZ);
    volScalarField S33 = S.component(tensor::ZZ);


    volScalarField u_ = this->U_.component(vector::X);
    volScalarField v_ = this->U_.component(vector::Y);
    volScalarField w_ = this->U_.component(vector::Z);

    volScalarField filter_width(this->delta());

    float input_vals[cellNum][in_channels];
    //torch::List<torch::Tensor> input_vals;
    const std::vector<std::int64_t> input_dims = {cellNum, in_channels};

    forAll(S11.internalField(), id) 
    {
        input_vals[id][0] = S11[id];
        input_vals[id][1] = S12[id];
        input_vals[id][2] = S13[id];    
        input_vals[id][3] = S22[id];
        input_vals[id][4] = S23[id];
        input_vals[id][5] = S33[id];
        input_vals[id][6] = u_[id];
        input_vals[id][7] = v_[id];
        input_vals[id][8] = w_[id];
    }
    

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .requires_grad(true);

    torch::jit::script::Module torch_module_;
    torch_module_ = torch::jit::load("/DNNLES/outputs/model.pt"); 
    torch::Tensor inputs = torch::from_blob(input_vals, {cellNum,in_channels}, options);

    std::vector<torch::jit::IValue> input_tensor;    
    input_tensor.push_back(inputs);

    torch::Tensor pred_Cs_ = torch_module_.forward(input_tensor).toTensor();

    for (int i = 0; i < cellNum; i++)
    {
        this->Cs_[i] = pred_Cs_[out_channels*i].item<float>();
    }

    this->Cs_ = filter_(this->Cs_);

    correctNut(gradU);
    
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //