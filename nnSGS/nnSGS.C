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

#include <torch/torch.h>
#include "nnSGS.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "model.H"
#include "CustomDataset.H"
#include <vector>

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

    LESeddyViscosity<BasicTurbulenceModel>::correct();

    const int64_t in_s = 9;
    const int64_t hd_s = 63; //256;
    const int64_t ot_s = 1;
    torch::DeviceType device = torch::kCPU; //kCUDA;

    int num_inputs = 9;
    int num_outputs = 1;

    float means [in_s+ot_s] = {1.000013622185760065e+00
                            ,-3.242322583163491140e-05
                            ,-2.501091846949404264e-05
                            ,-2.984414432006399741e-08
                            ,-1.814926145888257167e-05
                            ,9.966951957900526183e-06
                            ,1.491419986230899222e-08
                            ,1.012929592423972365e-05
                            ,1.488477140853993890e-08
                            ,2.107347679276886342e-04};
    float stds  [in_s+ot_s] = {5.506809586499618325e-02
                            ,1.734005998353918901e-02
                            ,1.735772506395298967e-02
                            ,6.644725533969407516e-02
                            ,7.958981269115324164e-02
                            ,7.998241736245030598e-02
                            ,5.677093472276837499e-02
                            ,4.532058630650669800e-02
                            ,5.658769966930470713e-02
                            ,7.237815949612259880e-02};


    newNN model(in_s, hd_s, ot_s);
    torch::load(model, "../nnTraining/best_model_052323.pt");

    volScalarField u_ = this->U_.component(vector::X);
    volScalarField v_ = this->U_.component(vector::Y);
    volScalarField w_ = this->U_.component(vector::Z);

    tmp<volTensorField> tgradU(fvc::grad(this->U_));
    const volTensorField& gradU = tgradU();
    volSymmTensorField S(dev(symm(gradU)));
    volScalarField S11 = S.component(tensor::XX);
    volScalarField S12 = S.component(tensor::XY);
    volScalarField S13 = S.component(tensor::XZ);
    volScalarField S22 = S.component(tensor::YY);
    volScalarField S23 = S.component(tensor::YZ);
    volScalarField S33 = S.component(tensor::ZZ);

    std::vector<std::vector<double>> in_data;
    forAll(u_, i)
    {
        std::vector<double> tmp;
        tmp.push_back((u_[i]-means[0])/stds[0]);
        tmp.push_back((v_[i]-means[1])/stds[1]);
        tmp.push_back((w_[i]-means[2])/stds[2]);
        tmp.push_back((S11[i]-means[3])/stds[3]);
        tmp.push_back((S12[i]-means[4])/stds[4]);
        tmp.push_back((S13[i]-means[5])/stds[5]);
        tmp.push_back((S22[i]-means[6])/stds[6]);
        tmp.push_back((S23[i]-means[7])/stds[7]);
        tmp.push_back((S33[i]-means[8])/stds[8]);

        in_data.push_back(tmp);
    }
    //std::cout << u_[0] << v_[0] << w_[0] << S11[0] << S12[0] << S13[0] << S22[0] << S23[0] << S33[0] << std::endl;
    //std::cout << "+--- in_data: " << in_data[0] << std::endl;
    const int64_t batchSize = in_data.size();
    Info << "+--- batch size: " << batchSize << nl;
    auto feat_ds  = CustomDataset(in_data, in_s).map(torch::data::transforms::Stack<>());
    auto dsloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(feat_ds), batchSize);
    Info << "+--- data loader is ready." << nl;

    model->to(device);
    Info << "+--- nn model is loaded to device." << nl;
    for(torch::data::Example<>& batch : *dsloader)
    {
        auto feat = batch.data.to(device).to(torch::kFloat32);
        //std::cout << "+--- feat: " << feat[0] << std::endl;
        auto pred = model->forward(feat);
        forAll(this->Cs_, i)
        {
            this->Cs_[i] = pred[i].item<float>() * stds[9] + means[9];
        }
    }
    
    this->Cs_ = filter_(this->Cs_);

    correctNut(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //