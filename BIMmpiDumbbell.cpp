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
    
    BIMobjects dumbbell(initPts, 12);    // no. of vertices of icosahedron = 12.
    
    dumbbell.refineMesh(3);

    if(myRank==0)   cout<<"number of elements: "<<dumbbell.getElementSize()<<endl;

    dumbbell.dumbbellTransform(1.0, 0.3);

    dumbbell.rotate(ThreeDVector(1.0, 0.0, 0.0), M_PI/4.0);
    
    if(myRank==0)    dumbbell.storeElemDat();
    
    //determine workloads for each core:
    int workloadVCalc[worldSize];
    for (int i = 0; i < worldSize; i++)
    {
        workloadVCalc[i] = dumbbell.nCoordFlat/worldSize;
        if(i < dumbbell.nCoordFlat%worldSize) workloadVCalc[i]++; // take care of remainders.
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
    
    dumbbell.refreshuS();
    for (int iter = 0; iter < 1; iter++)
    {
        for (int iGC = 0; iGC < dumbbell.nCoordFlat; iGC++)
        {
            dumbbell.picardIterate(iGC, myStartGC, myEndGC);
        }
        // get correct uSNxt for all processes. 
        for (int iGC = 0; iGC < dumbbell.nCoordFlat; iGC++)
        {
            double senduSNxt[3] = {dumbbell.uSNxt[iGC].x[0], dumbbell.uSNxt[iGC].x[1], dumbbell.uSNxt[iGC].x[2]};           
            double getuSNxt[3] = {dumbbell.uSNxt[iGC].x[0], dumbbell.uSNxt[iGC].x[1], dumbbell.uSNxt[iGC].x[2]};
                    
            MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            dumbbell.uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
        }
        dumbbell.refreshuS();

        if(myRank==0)    cout<<"iteration no:"<<iter+1<<endl;
        if(myRank==0)    cout<<dumbbell.uS[0].x[0]<<", "<<dumbbell.uS[0].x[1]<<", "<<dumbbell.uS[0].x[2]<<endl;
    }
    dumbbell.projectRB(); 

    dumbbell.uRB = dumbbell.integrateVectorfunc(&BIMobjects::getURB);
    dumbbell.omega = dumbbell.integrateVectorfunc(&BIMobjects::getOmegaRB);
    
    if(myRank==0)    cout<<"dumbbellURB: "<<dumbbell.uRB.x[0]<<", "<<dumbbell.uRB.x[1]<<", "<<dumbbell.uRB.x[2]<<endl;
    if(myRank==0)    cout<<"dumbbellomega: "<<dumbbell.omega.x[0]<<", "<<dumbbell.omega.x[1]<<", "<<dumbbell.omega.x[2]<<endl;
    
    double endTime = MPI_Wtime();
    if(myRank==0)   cout<<"time taken: "<<(endTime-startTime)<<endl;
    
    MPI_Finalize();

    return 0;
}