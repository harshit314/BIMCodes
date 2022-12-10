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
    
    BIMobjects sphereTemplate(initPts, 12);    // no. of vertices of icosahedron = 12.
    
    sphereTemplate.refineMesh(3);

    vector<BIMobjects> spheres;
    
    spheres.push_back(sphereTemplate);
    spheres.push_back(sphereTemplate);
    

    if(myRank==0)   cout<<"number of elements: "<<spheres[0].getElementSize()<<endl;

    spheres[1].translate(ThreeDVector(4.0, 0.0, 0.0) );
    
    if(myRank==0)   { spheres[0].storeElemDat(); }
    
    //determine workloads for each core:
    int workloadVCalc[worldSize];
    for (int i = 0; i < worldSize; i++)
    {
        workloadVCalc[i] = spheres[0].nCoordFlat/worldSize; //nCoordFlat doesnt change for different objects made out of sphere.
        if(i < spheres[0].nCoordFlat%worldSize) workloadVCalc[i]++; // take care of remainders.
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
    for (int iObj = 0; iObj < spheres.size(); iObj++)
    {
        spheres[iObj].refreshuS();
        // set closestGIndx for each pair of BIMobjects:
        spheres[iObj].setClosestGIndx(spheres, iObj);
    }
    if(myRank==0)    cout<<"Done. Starting Iterations..."<<endl;
    
    double startTime = MPI_Wtime();
    
    for (int iter = 0; iter < 10; iter++)
    {
        for (int iObj = 0; iObj < spheres.size(); iObj++)
        {
            spheres[iObj].resetUsNxt();
            for (int iGC = myStartGC; iGC < myEndGC; iGC++)
            {
                spheres[iObj].picardIterate(iGC, spheres, iObj);
            }
            // get correct uSNxt for all processes. 
            for (int iGC = 0; iGC < spheres[iObj].nCoordFlat; iGC++)
            {
                double senduSNxt[3] = {spheres[iObj].uSNxt[iGC].x[0], spheres[iObj].uSNxt[iGC].x[1], spheres[iObj].uSNxt[iGC].x[2]};           
                double getuSNxt[3] = {spheres[iObj].uSNxt[iGC].x[0], spheres[iObj].uSNxt[iGC].x[1], spheres[iObj].uSNxt[iGC].x[2]};
                        
                MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                spheres[iObj].uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
            }
            spheres[iObj].refreshuS();  //use new values as soon as you get them.
        }
        
        if(myRank==0)    cout<<"iteration no:"<<iter+1<<endl;
        if(myRank==0)    cout<<spheres[0].uS[0].x[0]<<", "<<spheres[0].uS[0].x[1]<<", "<<spheres[0].uS[0].x[2]<<endl;
    }
    
    for (int iObj = 0; iObj < spheres.size(); iObj++)
    {
        spheres[iObj].projectRB(); 

        spheres[iObj].uRB = spheres[iObj].integrateVectorfunc(&BIMobjects::getURB);
        spheres[iObj].omega = spheres[iObj].integrateVectorfunc(&BIMobjects::getOmegaRB);
    }    
    
    if(myRank==0)    cout<<"spheres[0]URB: "<<spheres[0].uRB.x[0]<<", "<<spheres[0].uRB.x[1]<<", "<<spheres[0].uRB.x[2]<<endl;
    if(myRank==0)    cout<<"spheres[0]omega: "<<spheres[0].omega.x[0]<<", "<<spheres[0].omega.x[1]<<", "<<spheres[0].omega.x[2]<<endl;
    
    if(myRank==0)    cout<<"spheres[1]URB: "<<spheres[1].uRB.x[0]<<", "<<spheres[1].uRB.x[1]<<", "<<spheres[1].uRB.x[2]<<endl;
    if(myRank==0)    cout<<"spheres[1]omega: "<<spheres[1].omega.x[0]<<", "<<spheres[1].omega.x[1]<<", "<<spheres[1].omega.x[2]<<endl;
    
    double endTime = MPI_Wtime();
    if(myRank==0)   cout<<"time taken: "<<(endTime-startTime)<<endl;
    
    MPI_Finalize();

    return 0;
}