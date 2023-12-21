#include "argList.H"
#include "timeSelector.H"
#include "ReadFields.H"
#include "volFields.H"
#include "IFstream.H"
#include "OFstream.H"
#include "fvc.H"

#include "fvCFD.H"

#include "interpolation.H"
#include "interpolationCellPoint.H"

// c++ include
#include <vector>

using namespace Foam;


template<class Type>
Type createField( 
    Foam::Time &runTime, 
    Foam::fvMesh &mesh,
    const std::string &fieldName
)
{
    Type field
    (
      IOobject
      (
        fieldName,
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
      ),
      mesh
    );

    return field;
}

template<class Type>
void collectFields(Field<Type>& field)
{
  List<Field<Type> > allEddies(Pstream::nProcs());
  
  allEddies[Pstream::myProcNo()] = field;
  
  Pstream::gatherList(allEddies);
  Pstream::scatterList(allEddies);
  
  field =
    ListListOps::combine<Field<Type> >
    (
     allEddies, 
     accessOp<Field<Type> >()
     );
}

int main(int argc, char* argv[])
{  
/*
    argList::validArgs.append("dsOption");
    argList::addOption
    (
      "dsOption",
      "number",
      "Specify option: 1 for USij 2 for UUp"
    );
*/    
    timeSelector::addOptions();

    Foam::argList args(argc, argv);
    if (!args.checkRootCase())
    {
      Foam::FatalError.exit();
    }
    //#include "setRootCase.H"
    #include "createTime.H" 
    #include "createNamedMesh.H"


    fileName outPath = mesh.time().path()/"postProcessing";
    mkDir(outPath);
	  autoPtr<OFstream> outPtr;
    outPtr.reset(new OFstream(outPath/"fieldData_rest.dat"));

    outPtr() << "t\t" 
             << "Ux\t"   << "Uy\t"   << "Uz\t"
             << "G1\t"   << "G2\t"   << "G3\t"   << "G4\t"   << "G5\t"   << "G6\t"
             << "S1\t"   << "S2\t"   << "S3\t"   << "S4\t"   << "S5\t"   << "S6\t"  
             << "UUp1\t" << "UUp2\t" << "UUp3\t" << "UUp4\t" << "UUp5\t" << "UUp6\t" 
             << "Cs" << endl;
    
    //while(runTime.loop())
    //{
    instantList timeDirs = timeSelector::select0(runTime, args);
    forAll(timeDirs, timeI)
    {
      runTime.setTime(timeDirs[timeI], timeI);
      if (runTime.timeName() != "0")
      {
        Info << runTime.timeName() << endl;
        volVectorField U = createField<volVectorField>(runTime, mesh, "U"); 
        volSymmTensorField UUp = createField<volSymmTensorField>(runTime,mesh,"UPrime2Mean");
        volSymmTensorField Sij = createField<volSymmTensorField>(runTime, mesh, "S_ij"); 
        volScalarField Cs = createField<volScalarField>(runTime, mesh, "Cs"); 
        //collectFields(U);
        volTensorField G = fvc::grad(U);
        volVectorField omega = fvc::curl(U);
        volSymmTensorField S(symm(G));  
        volScalarField Z = magSqr(omega);
        volScalarField SS = magSqr(S);
        
        //forAll(nInternalFaces, i)
        //forAll(U, i)
        //{
        //  outPtr() << U[i] << endl;
        //}
      
        
        for (label cellI = 0; cellI < mesh.C().size(); cellI++)
        {          
          if(cellI % 21 == 0)
          {
            outPtr()<< runTime.timeName() << "\t"
                    //<< mesh.C()[cellI][0] << "\t" << mesh.C()[cellI][1] << "\t" << mesh.C()[cellI][2] << "\t" 
                    << U  [cellI][0] << "\t" << U  [cellI][1] << "\t" << U  [cellI][2] << "\t"
                    << G  [cellI][0] << "\t" << G  [cellI][1] << "\t" << G  [cellI][2] << "\t"
                    << G  [cellI][3] << "\t" << G  [cellI][4] << "\t" << G  [cellI][5] << "\t"
                    << S  [cellI][0] << "\t" << S  [cellI][1] << "\t" << S  [cellI][2] << "\t"
                    << S  [cellI][3] << "\t" << S  [cellI][4] << "\t" << S  [cellI][5] << "\t" 
                    << UUp[cellI][0] << "\t" << UUp[cellI][1] << "\t" << UUp[cellI][2] << "\t" 
                    << UUp[cellI][3] << "\t" << UUp[cellI][4] << "\t" << UUp[cellI][5] << "\t" 
                    << Cs[cellI] << endl;      
          }
        }
      }
    }

    return 0;

}