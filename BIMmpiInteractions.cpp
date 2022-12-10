//BIMcurvedLinU contains all necessary header files and variables.
#include"./Headers/BIMcurvedLinUinteractions.h"
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
    
    BIMobjects spheroidTemplate(initPts, 12);    // no. of vertices of icosahedron = 12.
    
    spheroidTemplate.refineMesh(3);

    spheroidTemplate.scale(0.124656, 1.0, 0.124656);

    vector<BIMobjects> spheroids;
    
    spheroids.push_back(spheroidTemplate);

    spheroidTemplate.translate(ThreeDVector(4.0, 0.0, 0.0));

    spheroids.push_back(spheroidTemplate);
    

    if(myRank==0)   cout<<"number of elements: "<<spheroids[0].getElementSize()<<endl;
    
    if(myRank==0)   { spheroids[0].storeElemDat(); }
    
    //determine workloads for each core:
    int workloadVCalc[worldSize];
    for (int i = 0; i < worldSize; i++)
    {
        workloadVCalc[i] = spheroids[0].nCoordFlat/worldSize; //nCoordFlat doesnt change for different objects made out of sphere.
        if(i < spheroids[0].nCoordFlat%worldSize) workloadVCalc[i]++; // take care of remainders.
    }

    myStartGC = 0;
    for (int i = 0; i < myRank; i++)
    {
        myStartGC += workloadVCalc[i];
    }
    myEndGC = myStartGC + workloadVCalc[myRank];

    cout<<"myRank:"<<myRank<<" myStartGC:"<<myStartGC<<" myEndGC:"<<myEndGC<<endl;

    //*************//
    
    if(myRank==0)    cout<<"Assigning closest Indices..."<<endl;
    for (int iObj = 0; iObj < spheroids.size(); iObj++)
    {
        spheroids[iObj].refreshuS();
        // set closestGIndx for each pair of BIMobjects:
        spheroids[iObj].setClosestGIndx(spheroids, iObj);
    }
    if(myRank==0)    cout<<"Done. Starting Iterations..."<<endl;
    
    double startTime = MPI_Wtime();
    
    for (int iter = 0; iter < 10; iter++)
    {
        for (int iObj = 0; iObj < spheroids.size(); iObj++)
        {
            spheroids[iObj].resetUsNxt();
            for (int iGC = myStartGC; iGC < myEndGC; iGC++)
            {
                spheroids[iObj].picardIterate(iGC, spheroids, iObj);
            }
            // get correct uSNxt for all processes. 
            for (int iGC = 0; iGC < spheroids[iObj].nCoordFlat; iGC++)
            {
                double senduSNxt[3] = {spheroids[iObj].uSNxt[iGC].x[0], spheroids[iObj].uSNxt[iGC].x[1], spheroids[iObj].uSNxt[iGC].x[2]};           
                double getuSNxt[3] = {spheroids[iObj].uSNxt[iGC].x[0], spheroids[iObj].uSNxt[iGC].x[1], spheroids[iObj].uSNxt[iGC].x[2]};
                        
                MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                spheroids[iObj].uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
            }
            spheroids[iObj].refreshuS();  //use new values as soon as you get them.
        }
        
        if(myRank==0)    cout<<"iteration no:"<<iter+1<<endl;
        if(myRank==0)    cout<<spheroids[0].uS[0].x[0]<<", "<<spheroids[0].uS[0].x[1]<<", "<<spheroids[0].uS[0].x[2]<<endl;
    }
    
    for (int iObj = 0; iObj < spheroids.size(); iObj++)
    {
        spheroids[iObj].projectRB(); 

        spheroids[iObj].uRB = spheroids[iObj].integrateVectorfunc(&BIMobjects::getURB);
        spheroids[iObj].omega = spheroids[iObj].integrateVectorfunc(&BIMobjects::getOmegaRB);
    }    
    
    if(myRank==0)    cout<<"spheroids[0]URB: "<<spheroids[0].uRB.x[0]<<", "<<spheroids[0].uRB.x[1]<<", "<<spheroids[0].uRB.x[2]<<endl;
    if(myRank==0)    cout<<"spheroids[0]omega: "<<spheroids[0].omega.x[0]<<", "<<spheroids[0].omega.x[1]<<", "<<spheroids[0].omega.x[2]<<endl;
    
    if(myRank==0)    cout<<"spheroids[1]URB: "<<spheroids[1].uRB.x[0]<<", "<<spheroids[1].uRB.x[1]<<", "<<spheroids[1].uRB.x[2]<<endl;
    if(myRank==0)    cout<<"spheroids[1]omega: "<<spheroids[1].omega.x[0]<<", "<<spheroids[1].omega.x[1]<<", "<<spheroids[1].omega.x[2]<<endl;
    
    double endTime = MPI_Wtime();
    if(myRank==0)   cout<<"time taken: "<<(endTime-startTime)<<endl;
    
    MPI_Finalize();

    return 0;
}