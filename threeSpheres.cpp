//BIMcurvedLinU contains all necessary header files and variables.
#include"./Headers/BIMcurvedLinUinteractions.h"
#include<mpi/mpi.h>

double phi = 1.6180339887499;   //golden ratio
double dt = 0.02;

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

    sphereTemplate.translate(ThreeDVector(4.50, 0.0, 0.0));
    spheres.push_back(sphereTemplate);

    sphereTemplate.translate(ThreeDVector(4.50, 0.0, 0.0));
    spheres.push_back(sphereTemplate);

    //define orientation of the two disks:
    vector<ThreeDVector> dOrient;
    dOrient.push_back(ThreeDVector(1.0, 0.0, 0.0));
    dOrient.push_back(ThreeDVector(1.0, 0.0, 0.0));
    dOrient.push_back(ThreeDVector(1.0, 0.0, 0.0));

    if(myRank==0)   cout<<"number of elements: "<<spheres[0].getElementSize()<<endl;
    
   // if(myRank==0)   { spheres[0].storeElemDat(); }
    
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
    
    //******** write position and velocities of disks to a file***********//
    ofstream outPos, outVel;
    if(myRank==0)
    {
        outPos.open("./OutputData/posSphere.txt", ios::out);
        outPos.precision(nPrecision);
        outVel.open("./OutputData/velSphere.txt", ios::out);
        outVel.precision(nPrecision);
        for (int iObj = 0; iObj < spheres.size(); iObj++)
        {
            outPos<<spheres[iObj].x0.x[0]<<'\t'<<spheres[iObj].x0.x[1]<<'\t'<<spheres[iObj].x0.x[2]<<'\t';       
            outPos<<dOrient[iObj].x[0]<<'\t'<<dOrient[iObj].x[1]<<'\t'<<dOrient[iObj].x[2]<<'\t';       
        }    
        outPos<<'\n';
    }

    ThreeDVector lastuS;
    for (int iEvol = 0; iEvol < 2000; iEvol++)
    {
        for (int iter = 0; iter < 250; iter++)
        {
            double totalError = 0.0;
            double error = 0.0;
            for (int iObj = 0; iObj < spheres.size(); iObj++)
            {
                spheres[iObj].resetUsNxt();//all cpus start with zeroes and fill only there workloads.
                for (int iGC = myStartGC; iGC < myEndGC; iGC++)
                {
                    error += spheres[iObj].picardIterate(iGC, spheres, iObj);
                }
                // get correct uSNxt for all processes. 
                for (int iGC = 0; iGC < spheres[iObj].nCoordFlat; iGC++)
                {
                    double senduSNxt[3] = {spheres[iObj].uSNxt[iGC].x[0], spheres[iObj].uSNxt[iGC].x[1], spheres[iObj].uSNxt[iGC].x[2]};           
                    double getuSNxt[3] = {0.0, 0.0, 0.0};
                        
                    MPI_Allreduce(&senduSNxt, &getuSNxt, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    spheres[iObj].uSNxt[iGC].set(getuSNxt[0], getuSNxt[1], getuSNxt[2]);
                }
                
                MPI_Allreduce(&error, &totalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                spheres[iObj].refreshuS();  //use new values as soon as you get them.
            }
            
            if(myRank==0 && iter%10==0)    cout<<"iteration no:"<<iter+1<<"; totalError: "<<totalError<<endl;
            if( totalError <= 0.0001 )  break;
        }
        
        for (int iObj = 0; iObj < spheres.size(); iObj++)
        {
            spheres[iObj].projectRB(); 

            spheres[iObj].uRB = spheres[iObj].integrateVectorfunc(&BIMobjects::getURB);
            spheres[iObj].omega = spheres[iObj].integrateVectorfunc(&BIMobjects::getOmegaRB);

            //Euler method to evolve bodies:
            spheres[iObj].translate(spheres[iObj].uRB*dt);
            spheres[iObj].rotate(spheres[iObj].omega, dt); // magnitude of nHat contributes to rotation rate.
            dOrient[iObj] = dOrient[iObj].rotate(spheres[iObj].omega, dt);

            //store new Position in a file:
            if(myRank==0)    
            {
                outPos<<spheres[iObj].x0.x[0]<<'\t'<<spheres[iObj].x0.x[1]<<'\t'<<spheres[iObj].x0.x[2]<<'\t';       
                outPos<<dOrient[iObj].x[0]<<'\t'<<dOrient[iObj].x[1]<<'\t'<<dOrient[iObj].x[2]<<'\t';       
                //note down the calculated velocities at previous locations.
                outVel<<spheres[iObj].uRB.x[0]<<'\t'<<spheres[iObj].uRB.x[1]<<'\t'<<spheres[iObj].uRB.x[2]<<'\t';
                outVel<<spheres[iObj].omega.x[0]<<'\t'<<spheres[iObj].omega.x[1]<<'\t'<<spheres[iObj].omega.x[2]<<'\t';
            }
        }
        if(myRank==0)    
        {
            outPos<<'\n';
            outVel<<'\n';
        }
        for (int iObj = 0; iObj < spheres.size(); iObj++)
        {
            if(myRank==0)   cout<<"uRB and omega for next body:"<<endl;
            if(myRank==0)    cout<<"URB: "<<spheres[iObj].uRB.x[0]<<", "<<spheres[iObj].uRB.x[1]<<", "<<spheres[iObj].uRB.x[2]<<endl;
            if(myRank==0)    cout<<"Omega: "<<spheres[iObj].omega.x[0]<<", "<<spheres[iObj].omega.x[1]<<", "<<spheres[iObj].omega.x[2]<<endl;
        }
        //uSNxt holds last iterated value, set up Prb and P1 for next iterations now:
        for (int iObj = 0; iObj < spheres.size(); iObj++)
        {
            spheres[iObj].refreshuS();
            // set closestGIndx for each pair of BIMobjects:
            spheres[iObj].setClosestGIndx(spheres, iObj);
        }

        if(myRank==0)    cout<<"Evolved one step, begining evolution step:"<<iEvol+1<<endl;
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