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
#include <torch/script.h>
#include "SGS_NN.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "model.H"
#include "CustomDataset.H"
#include <vector>
#include <memory>
#include "string"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //
template<class BasicTurbulenceModel>
void SGS_NN<BasicTurbulenceModel>::correctNut
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
void SGS_NN<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SGS_NN<BasicTurbulenceModel>::SGS_NN
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
bool SGS_NN<BasicTurbulenceModel>::read()
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
void SGS_NN<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    LESeddyViscosity<BasicTurbulenceModel>::correct();

    Foam::IOdictionary nnProperties
    (
        Foam::IOobject
        (
            "nnProperties",
            Foam::fileName(this->runTime_.constant()),
            this->mesh_,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        )
    );

    int64_t MNum = Foam::readLabel(nnProperties.lookup("MNum"));

    const int ARRAY_SIZE = 22;
    std::vector<float> means(ARRAY_SIZE);
    std::vector<float> stds(ARRAY_SIZE);

    std::vector<std::string> keys = {"Ux","Uy","Uz",
                                        "G1","G2","G3","G4","G5","G6",
                                        "S1","S2","S3","S4","S5","S6",
                                        "UUp1","UUp2","UUp3","UUp4","UUp5","UUp6", 
                                        "Cs"};
    // Read values
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        means[i] = Foam::readScalar(nnProperties.subDict("means").lookup(Foam::word(keys[i])));
        stds[i] = Foam::readScalar(nnProperties.subDict("stds").lookup(Foam::word(keys[i])));
    }

    Info << "----**** means 0: " << means[0] << nl;
    Info << "----**** means 1: " << means[1] << nl;

    Info << "----**** stds 0: " << stds[0] << nl;
    Info << "----**** stds 1: " << stds[1] << nl;

    volScalarField u_ = this->U_.component(vector::X);
    volScalarField v_ = this->U_.component(vector::Y);
    volScalarField w_ = this->U_.component(vector::Z);

    volTensorField G = fvc::grad(this->U_);
    volScalarField G11 = G.component(tensor::XX);
    volScalarField G12 = G.component(tensor::XY);
    volScalarField G13 = G.component(tensor::XZ);
    volScalarField G22 = G.component(tensor::YY);
    volScalarField G23 = G.component(tensor::YZ);
    volScalarField G33 = G.component(tensor::ZZ);

    volSymmTensorField S(dev(symm(G)));
    volScalarField S11 = S.component(tensor::XX);
    volScalarField S12 = S.component(tensor::XY);
    volScalarField S13 = S.component(tensor::XZ);
    volScalarField S22 = S.component(tensor::YY);
    volScalarField S23 = S.component(tensor::YZ);
    volScalarField S33 = S.component(tensor::ZZ);

    volSymmTensorField UUp( IOobject("UPrime2Mean", this->runTime_.timeName(), this->mesh_, IOobject::READ_IF_PRESENT), symm(G));
    //volSymmTensorField UUp = createField<volSymmTensorField>(this->runTime_,this->mesh_,"UPrime2Mean");
    volScalarField UUp11 = UUp.component(tensor::XX);
    volScalarField UUp12 = UUp.component(tensor::XY);
    volScalarField UUp13 = UUp.component(tensor::XZ);
    volScalarField UUp22 = UUp.component(tensor::YY);
    volScalarField UUp23 = UUp.component(tensor::YZ);
    volScalarField UUp33 = UUp.component(tensor::ZZ);

    int64_t in_s = -3999;
    int64_t ot_s = -3999;
    
    std::vector<std::vector<double>> in_data;
    forAll(u_, i)
    {
        std::vector<double> tmp;
        if(MNum==1)
        {
            in_s = 9;
            ot_s = 1;
            tmp.push_back((u_[i]-means[0])/stds[0]);
            tmp.push_back((v_[i]-means[1])/stds[1]);
            tmp.push_back((w_[i]-means[2])/stds[2]);
            tmp.push_back((S11[i]-means[9])/stds[9]);
            tmp.push_back((S12[i]-means[10])/stds[10]);
            tmp.push_back((S13[i]-means[11])/stds[11]);
            tmp.push_back((S22[i]-means[12])/stds[12]);
            tmp.push_back((S23[i]-means[13])/stds[13]);
            tmp.push_back((S33[i]-means[14])/stds[14]);
        }
        else if(MNum==2)
        {
            in_s = 12;
            ot_s = 1;
            tmp.push_back((G11[i]-means[3])/stds[3]);
            tmp.push_back((G12[i]-means[4])/stds[4]);
            tmp.push_back((G13[i]-means[5])/stds[5]);
            tmp.push_back((G22[i]-means[6])/stds[6]);
            tmp.push_back((G23[i]-means[7])/stds[7]);
            tmp.push_back((G33[i]-means[8])/stds[8]);
            tmp.push_back((S11[i]-means[9])/stds[9]);
            tmp.push_back((S12[i]-means[10])/stds[10]);
            tmp.push_back((S13[i]-means[11])/stds[11]);
            tmp.push_back((S22[i]-means[12])/stds[12]);
            tmp.push_back((S23[i]-means[13])/stds[13]);
            tmp.push_back((S33[i]-means[14])/stds[14]);
        }
        else if(MNum==3)
        {
            in_s = 9;
            ot_s = 1;
            tmp.push_back((u_[i]-means[0])/stds[0]);
            tmp.push_back((v_[i]-means[1])/stds[1]);
            tmp.push_back((w_[i]-means[2])/stds[2]);
            tmp.push_back((UUp11[i]-means[15])/stds[15]);
            tmp.push_back((UUp12[i]-means[16])/stds[16]);
            tmp.push_back((UUp13[i]-means[17])/stds[17]);
            tmp.push_back((UUp22[i]-means[18])/stds[18]);
            tmp.push_back((UUp23[i]-means[19])/stds[19]);
            tmp.push_back((UUp33[i]-means[20])/stds[20]);
        }
        else if(MNum==4)
        {
            in_s = 12;
            ot_s = 1;
            tmp.push_back((G11[i]-means[3])/stds[3]);
            tmp.push_back((G12[i]-means[4])/stds[4]);
            tmp.push_back((G13[i]-means[5])/stds[5]);
            tmp.push_back((G22[i]-means[6])/stds[6]);
            tmp.push_back((G23[i]-means[7])/stds[7]);
            tmp.push_back((G33[i]-means[8])/stds[8]);
            tmp.push_back((UUp11[i]-means[15])/stds[15]);
            tmp.push_back((UUp12[i]-means[16])/stds[16]);
            tmp.push_back((UUp13[i]-means[17])/stds[17]);
            tmp.push_back((UUp22[i]-means[18])/stds[18]);
            tmp.push_back((UUp23[i]-means[19])/stds[19]);
            tmp.push_back((UUp33[i]-means[20])/stds[20]);
        }
        else if(MNum==5)
        {
            in_s = 21;
            ot_s = 1;
            tmp.push_back((u_[i]-means[0])/stds[0]);
            tmp.push_back((v_[i]-means[1])/stds[1]);
            tmp.push_back((w_[i]-means[2])/stds[2]);
            tmp.push_back((G11[i]-means[3])/stds[3]);
            tmp.push_back((G12[i]-means[4])/stds[4]);
            tmp.push_back((G13[i]-means[5])/stds[5]);
            tmp.push_back((G22[i]-means[6])/stds[6]);
            tmp.push_back((G23[i]-means[7])/stds[7]);
            tmp.push_back((G33[i]-means[8])/stds[8]);
            tmp.push_back((S11[i]-means[9])/stds[9]);
            tmp.push_back((S12[i]-means[10])/stds[10]);
            tmp.push_back((S13[i]-means[11])/stds[11]);
            tmp.push_back((S22[i]-means[12])/stds[12]);
            tmp.push_back((S23[i]-means[13])/stds[13]);
            tmp.push_back((S33[i]-means[14])/stds[14]);
            tmp.push_back((UUp11[i]-means[15])/stds[15]);
            tmp.push_back((UUp12[i]-means[16])/stds[16]);
            tmp.push_back((UUp13[i]-means[17])/stds[17]);
            tmp.push_back((UUp22[i]-means[18])/stds[18]);
            tmp.push_back((UUp23[i]-means[19])/stds[19]);
            tmp.push_back((UUp33[i]-means[20])/stds[20]);
        }

        in_data.push_back(tmp);
    }
    
    //std::cout << "+--- in_data: " << in_data[0] << std::endl;
    const int64_t batchSize = in_data.size();
    Info << "+--- batch size: " << batchSize << nl;
    auto ds  = CustomDataset(in_data, in_s).map(torch::data::transforms::Stack<>());
    auto dsloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(ds), batchSize);
    //Info << "+--- data loader is ready." << nl;

    std::string device = Foam::word(nnProperties.lookup("device"));
    Foam::fileName torchModelPath(nnProperties.lookup("torchModelPath"));

    torch::DeviceType deviceType;
    if (device == "cuda") 
    {
        deviceType = torch::kCUDA;
        Info << "+--- device is: " << device << nl;
    } 
    else if (device == "cpu")
    {
        deviceType = torch::kCPU;
        Info << "+--- device is: " << device << nl; 
    }
    
    torch::jit::script::Module torchModel = torch::jit::load(torchModelPath.c_str());
    torchModel.to(deviceType);
    torchModel.to(torch::kDouble);
    Info << "+--- torch model is loaded." << nl;
    for(torch::data::Example<>& batch : *dsloader)
    {
        auto feat = batch.data.to(device);
        c10::IValue ifeat = c10::IValue(feat); 
        //std::cout << "+--- feat: " << feat[0] << std::endl;
        auto output = torchModel.forward({ifeat});
        auto pred = output.toTensor();
        forAll(this->Cs_, i)
        {
            this->Cs_[i] = pred[i].item<float>() * stds[21] + means[21];
        }
    }
    
    this->Cs_ = filter_(this->Cs_);

    correctNut(G);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //