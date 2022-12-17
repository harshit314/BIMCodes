//BIMcurvedLinU contains all necessary header files and variables.
#include"./Headers/BIMcurvedLinUinteractions.h"
#include<mpi/mpi.h>

double phi = 1.6180339887499;   //golden ratio
double dt = 0.01;

double magn(ThreeDVector x, ThreeDVector x0, ThreeDVector DeluS, double w, vector<ThreeDVector> Itnsr)
{
    return DeluS.norm();
}

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

    spheroidTemplate.scale(0.124656, 1.0, 1.0);
    spheroidTemplate.rotate(ThreeDVector(0.0, 0.0, 1.0), M_PI/4.0);

    vector<BIMobjects> spheroids;
    
    spheroids.push_back(spheroidTemplate);

    spheroidTemplate.translate(ThreeDVector(3.0, 0.0, 0.0));
    spheroidTemplate.rotate(ThreeDVector(0.0, 0.0, 1.0), -M_PI/2.0);
    
    spheroids.push_back(spheroidTemplate);

    //define orientation of the two disks:
    vector<ThreeDVector> dOrient;
    dOrient.push_back(ThreeDVector(cos(M_PI/4.0), sin(M_PI/4.0), 0.0));
    dOrient.push_back(ThreeDVector(cos(M_PI/4.0), -sin(M_PI/4.0), 0.0));

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
    
    //******** write position and velocities of disks to a file***********//
    ofstream outPos, outVel;
    if(myRank==0)
    {
        outPos.open("./OutputData/pos.txt", ios::out);
        outPos.precision(nPrecision);
        outVel.open("./OutputData/vel.txt", ios::out);
        outVel.precision(nPrecision);
        for (int iObj = 0; iObj < spheroids.size(); iObj++)
        {
            outPos<<spheroids[iObj].x0.x[0]<<'\t'<<spheroids[iObj].x0.x[1]<<'\t'<<spheroids[iObj].x0.x[2]<<'\t';       
            outPos<<dOrient[iObj].x[0]<<'\t'<<dOrient[iObj].x[1]<<'\t'<<dOrient[iObj].x[2]<<'\t';       
        }    
        outPos<<'\n';
    }

    for (int iEvol = 0; iEvol < 1000; iEvol++)
    {
        for (int iter = 0; iter < 250; iter++)
        {
            double totalError = 0.0;
            double error = 0.0;
            for (int iObj = 0; iObj < spheroids.size(); iObj++)
            {
                spheroids[iObj].resetUsNxt();   //all cpus start with zeroes and fill only there workloads.
                for (int iGC = myStartGC; iGC < myEndGC; iGC++)
                {
                    error += spheroids[iObj].picardIterate(iGC, spheroids, iObj);
                }
                // get correct uSNxt for all processes. 
                for (int iGC = 0; iGC < spheroids[iObj].nCoordFlat; iGC++)
                {
                    double senduSNxt[3] = {spheroids[iObj].uSNxt[iGC].x[0], spheroids[iObj].uSNxt[iGC].x[1], spheroids[iObj].uSNxt[iGC].x[2]};           
                    double getuSNxt[3] = {0.0, 0.0, 0.0};
                            
                    MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    spheroids[iObj].uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
                }
                
                MPI_Allreduce(&error, &totalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);    
                spheroids[iObj].refreshuS();  //use new values as soon as you get them.
            }
            if(myRank==0 && iter%10==0)    cout<<"iteration no:"<<iter+1<<"; AvgError: "<<totalError/spheroids[0].nCoordFlat<<endl;
            if( totalError/spheroids[0].nCoordFlat <= 0.0001 )  break;
        }
        
        for (int iObj = 0; iObj < spheroids.size(); iObj++)
        {
            spheroids[iObj].projectRB(); 

            spheroids[iObj].uRB = spheroids[iObj].integrateVectorfunc(&BIMobjects::getURB);
            spheroids[iObj].omega = spheroids[iObj].integrateVectorfunc(&BIMobjects::getOmegaRB);

            //Euler method to evolve bodies:
            spheroids[iObj].translate(spheroids[iObj].uRB*dt);
            spheroids[iObj].rotate(spheroids[iObj].omega, dt); // magnitude of nHat contributes to rotation rate.
            dOrient[iObj] = dOrient[iObj].rotate(spheroids[iObj].omega, dt);

            //store new Position in a file:
            if(myRank==0)    
            {
                outPos<<spheroids[iObj].x0.x[0]<<'\t'<<spheroids[iObj].x0.x[1]<<'\t'<<spheroids[iObj].x0.x[2]<<'\t';       
                outPos<<dOrient[iObj].x[0]<<'\t'<<dOrient[iObj].x[1]<<'\t'<<dOrient[iObj].x[2]<<'\t';       
                //note down the calculated velocities at previous locations.
                outVel<<spheroids[iObj].uRB.x[0]<<'\t'<<spheroids[iObj].uRB.x[1]<<'\t'<<spheroids[iObj].uRB.x[2]<<'\t';
                outVel<<spheroids[iObj].omega.x[0]<<'\t'<<spheroids[iObj].omega.x[1]<<'\t'<<spheroids[iObj].omega.x[2]<<'\t';
            }
        }
        if(myRank==0)    
        {
            outPos<<'\n';
            outVel<<'\n';
        }
        for (int iObj = 0; iObj < spheroids.size(); iObj++)
        {
            if(myRank==0)   cout<<"uRB and omega for next body:"<<endl;
            if(myRank==0)    cout<<"URB: "<<spheroids[iObj].uRB.x[0]<<", "<<spheroids[iObj].uRB.x[1]<<", "<<spheroids[iObj].uRB.x[2]<<endl;
            if(myRank==0)    cout<<"Omega: "<<spheroids[iObj].omega.x[0]<<", "<<spheroids[iObj].omega.x[1]<<", "<<spheroids[iObj].omega.x[2]<<endl;
        }
        //uSNxt holds last iterated value, set up Prb and P1 for next iterations now:
        for (int iObj = 0; iObj < spheroids.size(); iObj++)
        {
            spheroids[iObj].refreshuS();
            // set closestGIndx for each pair of BIMobjects:
            spheroids[iObj].setClosestGIndx(spheroids, iObj);
        }

        if(myRank==0)    cout<<"Evolved one step, starting iterations no. "<<iEvol+1<<endl;
    }

    double endTime = MPI_Wtime();
    if(myRank==0)   cout<<"time taken: "<<(endTime-startTime)<<endl;
    
    MPI_Finalize();

    if(myRank==0)    
    {
        outPos.close();
        outVel.close();
    }
    return 0;
}