//BIMcurvedLinU contains all necessary header files and variables.
#include"./Headers/BIMcurvedLinU.h"
#include<mpi/mpi.h>

double phi = 1.6180339887499;   //golden ratio

int main(int argc, char **argv)
{
    cout.precision(nPrecision);
    
    //Initialize MPI:    
    //mpi variables.
    int worldSize, myRank, myStartGC, myEndGC;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    ThreeDVector initPts[] = {ThreeDVector(0, 1, phi), ThreeDVector(0, -1, phi), ThreeDVector(0, 1, -phi), ThreeDVector(0, -1, -phi),
                                ThreeDVector(1, phi, 0), ThreeDVector(-1, phi, 0), ThreeDVector(1, -phi, 0), ThreeDVector(-1, -phi, 0),
                                ThreeDVector(phi, 0, 1), ThreeDVector(-phi, 0, 1), ThreeDVector(phi, 0, -1), ThreeDVector(-phi, 0, -1)}; //vertices of icosahedron.
    
    BIMobjects spheroid(initPts, 12);    // no. of vertices of icosahedron = 12.
    
    spheroid.refineMesh(3);

    if(myRank==0)   cout<<"number of elements: "<<spheroid.getElementSize()<<endl;

    spheroid.scale(1.0, 0.714143, 0.714143);

    spheroid.rotate(ThreeDVector(0.0, 0.0, 1.0), M_PI/4.0);

    if(myRank==0)    spheroid.storeElemDat();
    
    //determine workloads for each core:
    int workloadVCalc[worldSize];
    for (int i = 0; i < worldSize; i++)
    {
        workloadVCalc[i] = spheroid.nCoordFlat/worldSize;
        if(i < spheroid.nCoordFlat%worldSize) workloadVCalc[i]++; // take care of remainders.
    }

    myStartGC = 0;
    for (int i = 0; i < myRank; i++)
    {
        myStartGC += workloadVCalc[i];
    }
    myEndGC = myStartGC + workloadVCalc[myRank];

    cout<<"myRank:"<<myRank<<" myStartGC:"<<myStartGC<<" myEndGC:"<<myEndGC<<endl;

    //*************//
    
    double startTime = MPI_Wtime();
    
    spheroid.refreshuS();
    for (int iter = 0; iter < 10; iter++)
    {
        spheroid.resetUsNxt();
        for (int iGC = myStartGC; iGC < myEndGC; iGC++)
        {
            spheroid.picardIterate(iGC);
        }
        // get correct uSNxt for all processes. 
        for (int iGC = 0; iGC < spheroid.nCoordFlat; iGC++)
        {
            double senduSNxt[3] = {spheroid.uSNxt[iGC].x[0], spheroid.uSNxt[iGC].x[1], spheroid.uSNxt[iGC].x[2]};           
            double getuSNxt[3] = {spheroid.uSNxt[iGC].x[0], spheroid.uSNxt[iGC].x[1], spheroid.uSNxt[iGC].x[2]};
                    
            MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            spheroid.uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
        }
        spheroid.refreshuS();

        if(myRank==0)    cout<<"iteration no:"<<iter+1<<endl;
        if(myRank==0)    cout<<spheroid.uS[0].x[0]<<", "<<spheroid.uS[0].x[1]<<", "<<spheroid.uS[0].x[2]<<endl;
    }
    spheroid.projectRB(); 

    spheroid.uRB = spheroid.integrateVectorfunc(&BIMobjects::getURB);
    spheroid.omega = spheroid.integrateVectorfunc(&BIMobjects::getOmegaRB);
    
    if(myRank==0)    cout<<"spheroidURB: "<<spheroid.uRB.x[0]<<", "<<spheroid.uRB.x[1]<<", "<<spheroid.uRB.x[2]<<endl;
    if(myRank==0)    cout<<"spheroidomega: "<<spheroid.omega.x[0]<<", "<<spheroid.omega.x[1]<<", "<<spheroid.omega.x[2]<<endl;
    
    double endTime = MPI_Wtime();
    if(myRank==0)   cout<<"time taken: "<<(endTime-startTime)<<endl;
    
    MPI_Finalize();

    return 0;
}