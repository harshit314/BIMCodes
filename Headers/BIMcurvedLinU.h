#include<iostream>
#include<iterator>
#include<math.h>
#include<map>
#include<vector>
#include<fstream>
#include<iomanip>   // use cout<<fixed after setting precision set by cout.precision(n) 

using namespace std;

// parameters:
ifstream quadratureDat;
int nPrecision = 14;

class ThreeDVector
{
    public:
    double x[3];
    ThreeDVector(){}
    ThreeDVector(double x[3])
    {
        for (int i = 0; i < 3; i++) this->x[i] = x[i];
    }
    ThreeDVector(double x, double y, double z)
    {
        double temp[3] = {x,y,z};
        for (int i = 0; i < 3; i++) this->x[i] = temp[i];
    }
    void set(double x, double y, double z)
    {
        double temp[3] = {x,y,z};
        for (int i = 0; i < 3; i++) this->x[i] = temp[i];
    }
    void set(ThreeDVector vec)
    {
        for (int i = 0; i < 3; i++) this->x[i] = vec.x[i];
    }
    double dot(ThreeDVector vec)
    {
        double value = 0.0;
        for (int i = 0; i < 3; i++) value += this->x[i]*vec.x[i];
        return value;
    }
    double dot(double vec[])
    {
        double value = 0.0;
        for (int i = 0; i < 3; i++) value += this->x[i]*vec[i];
        return value;
    }

    ThreeDVector cross(ThreeDVector vec)
    {
        ThreeDVector result;
        result.x[0] = this->x[1]*vec.x[2]-this->x[2]*vec.x[1];
        result.x[1] = this->x[2]*vec.x[0]-this->x[0]*vec.x[2];
        result.x[2] = this->x[0]*vec.x[1]-this->x[1]*vec.x[0];

        return result;
    }

    double norm()
    {
        double value = 0.0;
        for (int i = 0; i < 3; i++) value += this->x[i]*this->x[i];
        return sqrt(value);
    }

    ThreeDVector normalize()
    {
        double mag = this->norm();
        ThreeDVector res = this->x;
        return res*(1.0/mag);
    }

    ThreeDVector operator + (ThreeDVector vec)
    {
        double value[3];
        for (int i = 0; i < 3; i++) value[i] = this->x[i]+vec.x[i];
        return ThreeDVector(value);
    }

    ThreeDVector operator - (ThreeDVector vec)
    {
        double value[3];
        for (int i = 0; i < 3; i++) value[i] = this->x[i]-vec.x[i];
        return ThreeDVector(value);
    }
    ThreeDVector operator * (double c)
    {
        double value[3];
        for (int i = 0; i < 3; i++) value[i] = c*this->x[i];
        return ThreeDVector(value);
    }

    ThreeDVector rotate(ThreeDVector w, double delT)
    {  
        //Rotate r vector using w vector by an angle phi: dr/dt = wxr
        if(w.norm() == 0.0) return x;   //INCLUDE IN OLD CODES TO DEBUG
        ThreeDVector result;
        double wSq = w.dot(w), w1Sq = w.x[0]*w.x[0], w2Sq = w.x[1]*w.x[1], w3Sq = w.x[2]*w.x[2];
        double modW = w.norm(), phi = modW*delT, w1 = w.x[0], w2 = w.x[1], w3 = w.x[2];
        //see mathematica code for rotation matrix 
        result.x[0] = ( x[0]*( w1Sq + (w2Sq+w3Sq)*cos(phi) ) + x[2]*( w1*w3*(1-cos(phi)) + w2*modW*sin(phi) ) + x[1]*( w1*w2*(1-cos(phi)) - w3*modW*sin(phi) ) )/wSq;
        result.x[1] = ( x[1]*( w2Sq + (w1Sq+w3Sq)*cos(phi) ) + x[2]*( w2*w3*(1-cos(phi)) - w1*modW*sin(phi) ) + x[0]*( w1*w2*(1-cos(phi)) + w3*modW*sin(phi) ) )/wSq;
        result.x[2] = ( x[2]*( w3Sq + (w1Sq+w2Sq)*cos(phi) ) + x[1]*( w2*w3*(1-cos(phi)) + w1*modW*sin(phi) ) + x[0]*( w1*w3*(1-cos(phi)) - w2*modW*sin(phi) ) )/wSq;
        return result;
    }

};

class mesh
{   
    protected:
    //for each element: key labels the element; 3d vector stores global labels of three vertices that make up the element.
    map<int, ThreeDVector> globalCoord, element, nextElement, elementMid;    
    //store element indices in global index:
    map<int, vector<int> > elementsInGlobalIndx;    //elementsInGlobalIndx is a map, elementsInGlobalIndx[i] is a vector of int which has list of elements that contain the global index i. 
    // each iteration appends globalCoord but all elements need to be recalculated.
    
    // return the labels of 2-simplices which contain the 0-simplex denoted by input argument label:
    vector<int> whichElem(int label)   
    {
        vector<int> res;
        int count=0;
        for (int iElem = 0; iElem < element.size(); iElem++) //
        {
            for(int j=0; j<3; j++)  
            {
                if(element[iElem].x[j] == label)
                {
                    res.push_back(iElem);
                    count++;
                    break;
                } 
            }
        }
        if(res.empty()) cout<<"Can't find neighbouring 2-simplices for this 0-simplex!!"<<label<<endl;
        return res;
    }   
    
    // return the labels of 2-simplices which contain the 0-simplices denoted by input argument labels:
    vector<int> whichElem(int label1, int label2)  
    {
        vector<int> res;
        int count=0;
        if(label1 == label2)    cout<<"Both labels are same, can't find 2 2-simplices; will cause errors"<<endl;
        for (int jElem = 0; jElem < element.size(); jElem++) //
        {
            if(element[jElem].x[0] == label1 || element[jElem].x[1] == label1 || element[jElem].x[2] == label1)
            {
                if(element[jElem].x[0] == label2 || element[jElem].x[1] == label2 || element[jElem].x[2] == label2)    //make sure label1 and label2 are different!
                {
                    res.push_back(jElem);
                    count++;
                }
            } 
            if(count == 2)  break;  // 1-simplex can only contain 2 2-simplices.  
        }
        if(res.empty()) cout<<"Can't find 2 neighbouring 2-simplices for this 1-simplex!!"<<endl;
        return res;
    }
    
    int getOtherElemIndx(int * commonElemIndx, int currentElemIndx)
    {
        int otherElemLabel;
        otherElemLabel = commonElemIndx[0]==currentElemIndx?commonElemIndx[1]:commonElemIndx[0];   
        return otherElemLabel;
    }

    void updateMidPtsIndx(int otherElemIndx, int currElemIndx, int * mGIndx, map<int, ThreeDVector> midPtsGIndx, ThreeDVector mPt, int edgeGIndx1, int edgeGIndx2)
    {
        if(otherElemIndx < currElemIndx)   
        {
            int indx;
            //get local indx of other element
            if(element[otherElemIndx].x[0] == edgeGIndx1 || element[otherElemIndx].x[0] == edgeGIndx2)
            {
                if(element[otherElemIndx].x[1] == edgeGIndx1 || element[otherElemIndx].x[1] == edgeGIndx2)  indx = 0;
                else    indx = 2;
            }
            else    indx = 1;

            *mGIndx = midPtsGIndx[otherElemIndx].x[indx];   // compute indx as per the old element
        //    cout<<*mGIndx<<" nC:"<<nCoord<<" curr:"<<currElemIndx<<" othr:"<<otherElemIndx<<endl;
        }
        else
        {
            *mGIndx = nCoord;
            globalCoord[nCoord] = mPt; 
            nCoord = globalCoord.size();
        }
            
    }

    void refineMesh()
    {
        nextElement.clear();
        ThreeDVector m1, m2, m3, currElemGIndx; // get global indices of current element using currElemGIndx
        int nElem = 0; //count the current number of next elements
        
        map<int, ThreeDVector> midPtsGIndx; // defined for each element, contains global index of midPoints of each element

        for (int iElem = 0; iElem < element.size(); iElem++)
        {
            currElemGIndx = element[iElem]; //change begin() to generalize for all elements
            m1 = (globalCoord[currElemGIndx.x[0]] + globalCoord[currElemGIndx.x[1]])*0.5;
            m2 = (globalCoord[currElemGIndx.x[1]] + globalCoord[currElemGIndx.x[2]])*0.5;
            m3 = (globalCoord[currElemGIndx.x[2]] + globalCoord[currElemGIndx.x[0]])*0.5;
            //project midpoints on the surface of sphere:
            m1 = m1.normalize();
            m2 = m2.normalize();
            m3 = m3.normalize();

            //check for double counting:
            int otherElemIndx, m1GIndx, m2GIndx, m3GIndx;
            //for m1:
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[0], currElemGIndx.x[1]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m1GIndx, midPtsGIndx, m1, currElemGIndx.x[0], currElemGIndx.x[1]);
            
            //for m2:
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[1], currElemGIndx.x[2]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m2GIndx, midPtsGIndx, m2, currElemGIndx.x[1], currElemGIndx.x[2]);
            
            //for m3:
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[2], currElemGIndx.x[0]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m3GIndx, midPtsGIndx, m3, currElemGIndx.x[2], currElemGIndx.x[0]);
            
            //add new points to the global coordinates; old global coordinates should be unchanged:
            
            midPtsGIndx[iElem] = ThreeDVector(m1GIndx, m2GIndx, m3GIndx);  
            
            //update the element:
            ThreeDVector currElemMidGIndx = midPtsGIndx[iElem];

            nextElement[nElem] = ThreeDVector(currElemGIndx.x[0], currElemMidGIndx.x[0], currElemMidGIndx.x[2]);
            nextElement[nElem+1] = ThreeDVector(currElemMidGIndx.x[0], currElemMidGIndx.x[1], currElemMidGIndx.x[2]);
            nextElement[nElem+2] = ThreeDVector(currElemMidGIndx.x[0], currElemGIndx.x[1], currElemMidGIndx.x[1]);
            nextElement[nElem+3] = ThreeDVector(currElemMidGIndx.x[1], currElemGIndx.x[2], currElemMidGIndx.x[2]);

            //each iteration adds 4 elements to nextElement
            nElem += 4;

        }

        element.clear();
        element = nextElement;
        
    }

    void setElemMid()
    {
        elementMid.clear();
        ThreeDVector m1, m2, m3, currElemGIndx; // get global indices of current element using currElemGIndx
    
        map<int, ThreeDVector> midPtsGIndx; // defined for each element, contains global index of midPoints of each element

        for (int iElem = 0; iElem < element.size(); iElem++)
        {
            currElemGIndx = element[iElem]; //change begin() to generalize for all elements
            m1 = (globalCoord[currElemGIndx.x[0]] + globalCoord[currElemGIndx.x[1]])*0.5;
            m2 = (globalCoord[currElemGIndx.x[1]] + globalCoord[currElemGIndx.x[2]])*0.5;
            m3 = (globalCoord[currElemGIndx.x[2]] + globalCoord[currElemGIndx.x[0]])*0.5;
            //project midpoints on the surface of sphere:
            m1 = m1.normalize();
            m2 = m2.normalize();
            m3 = m3.normalize();

            //check for double counting:
            int otherElemIndx, m1GIndx, m2GIndx, m3GIndx;
            //for m1:
            // if the mid point is already counted in other element, get its global index; otherwise add it in globalCoord and update its global index
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[0], currElemGIndx.x[1]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m1GIndx, midPtsGIndx, m1, currElemGIndx.x[0], currElemGIndx.x[1]);
            
            //for m2:
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[1], currElemGIndx.x[2]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m2GIndx, midPtsGIndx, m2, currElemGIndx.x[1], currElemGIndx.x[2]);
            
            //for m3:
            otherElemIndx = getOtherElemIndx(whichElem(currElemGIndx.x[2], currElemGIndx.x[0]).data(), iElem);
            updateMidPtsIndx(otherElemIndx, iElem, &m3GIndx, midPtsGIndx, m3, currElemGIndx.x[2], currElemGIndx.x[0]);
            
            //add new points to the global coordinates; old global coordinates should be unchanged:
            
            midPtsGIndx[iElem] = ThreeDVector(m1GIndx, m2GIndx, m3GIndx);  
            
            //update the elementMid:
            elementMid[iElem] = midPtsGIndx[iElem];
        }

    }

    void setElementsInGlobalIndx()
    {
        elementsInGlobalIndx.clear();
        for (int iGC = 0; iGC < nCoordFlat; iGC++)
        {
            elementsInGlobalIndx[iGC] = whichElem(iGC);
        }
    }

    public:
    // number of global coordinates.
    int nCoord; 
    // nCoordFlat is no. of global elements when flat elements are used.
    int nCoordFlat; 
    // center of the mesh in 3d space.
    ThreeDVector x0;

    // size must be 12 to begin with an icosahedron
    mesh(ThreeDVector * pts, int size)  
    {
        for (int i = 0; i < size; i++)
        {
            globalCoord[i] = pts[i].normalize();
        }
        // for icosahedron, 20 elements are created at the beginning:
        element[0] = ThreeDVector(5, 0, 4); element[1] = ThreeDVector(5, 4, 2); element[2] = ThreeDVector(1, 7, 6); element[3] = ThreeDVector(6, 7, 3);
        
        element[4] = ThreeDVector(6, 10, 8); element[5] = ThreeDVector(10, 4, 8); element[6] = ThreeDVector(11, 9, 5); element[7] = ThreeDVector(11, 7, 9);
        
        element[8] = ThreeDVector(0, 9, 1); element[9] = ThreeDVector(0, 1, 8); element[10] = ThreeDVector(2, 10, 3); element[11] = ThreeDVector(2, 3, 11);
        
        element[12] = ThreeDVector(6, 8, 1); element[13] = ThreeDVector(9, 7, 1); element[14] = ThreeDVector(9, 0, 5); element[15] = ThreeDVector(0, 8, 4);
        
        element[16] = ThreeDVector(10, 6, 3); element[17] = ThreeDVector(3, 7, 11); element[18] = ThreeDVector(11, 5, 2); element[19] = ThreeDVector(4, 10, 2);
    
        nCoord = globalCoord.size();
        
        x0 = ThreeDVector(0.0, 0.0, 0.0);
    }

    void refineMesh(int nTimes)
    {
        for (int i = 0; i < nTimes; i++)
        {
            refineMesh();
        }
        //assign nCoordFlat BEFORE setElemMid:
        nCoordFlat = globalCoord.size();
        setElemMid();
        setElementsInGlobalIndx();
    }

    int getElementSize()    {  return element.size(); }

    void storeElemDat()
    {
        ofstream outFile;
        outFile.open("./meshData/spheroid4.txt", ios::out); //folder must be present!
        outFile.precision(nPrecision);
            
        for (int iElem = 0; iElem < element.size(); iElem++)
        {
            outFile<<fixed<<iElem<<'\t';
            for (int iPos = 0; iPos < 3; iPos++)
            {
                ThreeDVector pos = globalCoord[element[iElem].x[iPos]];
                outFile<<pos.x[0]<<"\t"<<pos.x[1]<<"\t"<<pos.x[2]<<"\t";
            }
            outFile<<'\n';
        }
    
    }

    //*********define functions to transform the mesh*************
    //remember to update x0 and area after each transformation!

    // make ellipsoid out of sphere:
    void scale(double a, double b, double c) 
    {
        for (int iGC = 0; iGC < globalCoord.size(); iGC++)
        {
            //scale each global coordinate:
            globalCoord[iGC].x[0] *= a;
            globalCoord[iGC].x[1] *= b;
            globalCoord[iGC].x[2] *= c;
        }
    }

    //rotate points by theta about nHat about x0:
    void rotate(ThreeDVector nHat, double theta) 
    {
        for (int iGC = 0; iGC < globalCoord.size(); iGC++)  
        {
            globalCoord[iGC] = globalCoord[iGC] - x0;
            globalCoord[iGC] = globalCoord[iGC].rotate(nHat, theta);
            globalCoord[iGC] = globalCoord[iGC] + x0;
        }
    }

    void translate(ThreeDVector dx)
    {
        for (int iGC = 0; iGC < globalCoord.size(); iGC++)   globalCoord[iGC] = globalCoord[iGC] + dx;
        x0 = x0 + dx;
    }


};

class BIMobjects: public mesh
{
    protected:
    ThreeDVector uRBAux, omegaAux;
    vector<ThreeDVector> Itensor, ItensorInv;

    vector<double> xi, eta, w; //quadrature points for all elements; assigned using quadratureDat file

    void getItensor()
    {
        Itensor.clear();    ItensorInv.clear();

        Itensor.push_back(integrateVectorfunc(&BIMobjects::Drow1));
        Itensor.push_back(integrateVectorfunc(&BIMobjects::Drow2));
        Itensor.push_back(integrateVectorfunc(&BIMobjects::Drow3));

        double detItensor = Itensor[0].dot( Itensor[1].cross(Itensor[2]) ); //det=scalar triple product   
        if(detItensor==0)   
        {
            cout<<"Itensor is singular!!"<<endl;
        }

        //get inverse of Itensor, a symmetric tensor:
        // [I[0].x[0]  I[0].x[1]   I[0].x[2]]
        // [I[1].x[0]  I[1].x[1]   I[1].x[2]]
        // [I[2].x[0]  I[2].x[1]   I[2].x[2]]
        double t11 = Itensor[1].x[1]*Itensor[2].x[2]-Itensor[1].x[2]*Itensor[2].x[1];
        double t12 = -(Itensor[1].x[0]*Itensor[2].x[2]-Itensor[1].x[2]*Itensor[2].x[0]);
        double t13 = Itensor[1].x[0]*Itensor[2].x[1]-Itensor[1].x[1]*Itensor[2].x[0];
        
        double t22 = Itensor[0].x[0]*Itensor[2].x[2]-Itensor[0].x[2]*Itensor[2].x[0];
        double t23 = -(Itensor[0].x[0]*Itensor[2].x[1]-Itensor[0].x[1]*Itensor[2].x[0]);
        double t33 = Itensor[0].x[0]*Itensor[1].x[1]-Itensor[1].x[0]*Itensor[0].x[1];

        ItensorInv.push_back(ThreeDVector(t11, t12, t13) * (1.0/detItensor));
        ItensorInv.push_back(ThreeDVector(t12, t22, t23) * (1.0/detItensor));
        ItensorInv.push_back(ThreeDVector(t13, t23, t33) * (1.0/detItensor));
    
    }

    void initializeUs()
    {
        uS.clear();
        uSNxt.clear();
        for (int iGC = 0; iGC < nCoordFlat; iGC++)
        {
            uS[iGC] = ThreeDVector(0.694, -1.693, -0.3079); // uS defined for each 3 nodes of an element.
            uSNxt[iGC] = ThreeDVector(0.694, -1.693, -0.3079);
        }
    }

    //correct singularities!!! xi, eta and zeta at 1/3 don't give x = xM! Saves the 1/0 evaluation!
    ThreeDVector integralDLOp(int eIndx, int GIndx)
    {
        ThreeDVector xPrime = globalCoord[GIndx];
        // ThreeDVector xBar = xPrime-x0;
        ThreeDVector res(0.0, 0.0, 0.0);
        for (int iElem = 0; iElem < element.size(); iElem++)
        {
           // if(iElem->first == eIndx)   continue;   //delU is zero anyway!
            //get global coordinates of the element:
            ThreeDVector x1 = globalCoord[element[iElem].x[0]].x;
            ThreeDVector x2 = globalCoord[element[iElem].x[1]].x;
            ThreeDVector x3 = globalCoord[element[iElem].x[2]].x;
            // get midPoints of elements:
            ThreeDVector x4 = globalCoord[elementMid[iElem].x[0]].x;
            ThreeDVector x5 = globalCoord[elementMid[iElem].x[1]].x;
            ThreeDVector x6 = globalCoord[elementMid[iElem].x[2]].x;
            
          //  ThreeDVector delU = uS[iElem] - uS[eIndx];

            // map to standard triangle:
            // general vector inside the element: x = Sum_{i=1}^6 x_i phi_i
            
            for(int iQuad=0; iQuad < w.size(); iQuad++)
            {
                double xiIn = xi[iQuad], etaIn = eta[iQuad], zetaIn = 1.0 - xi[iQuad] - eta[iQuad];
                
                //basis vectors:
                ThreeDVector eXi = x1*(1.0-4.0*zetaIn) + x2*(4.0*xiIn - 1.0) + x4*(4.0*zetaIn - 4.0*xiIn) + x5*(4.0*etaIn) - x6*(4.0*etaIn);
                ThreeDVector eEta = x1*(1.0-4.0*zetaIn) + x3*(4.0*etaIn - 1.0) - x4*(4.0*xiIn) + x5*(4.0*xiIn) + x6*(4.0*zetaIn - 4.0*etaIn);
                
                double hS = eXi.cross(eEta).norm();
                ThreeDVector normalVector = eXi.cross(eEta)*(1.0/hS);
                
                ThreeDVector xElem = x1*(zetaIn*(2.0*zetaIn-1.0)) + x2*(xiIn*(2.0*xiIn - 1.0)) + x3*(etaIn*(2.0*etaIn-1.0)) + x4*(4.0*zetaIn*xiIn) + x5*(4.0*xiIn*etaIn) + x6*(4.0*etaIn*zetaIn);
                
                //linear interpolation for uS:
                ThreeDVector uSElem = uS[element[iElem].x[0]]*(zetaIn) + uS[element[iElem].x[1]]*(xiIn) + uS[element[iElem].x[2]]*(etaIn);

                ThreeDVector delU = uSElem - uS[GIndx];

                ThreeDVector r = xElem - xPrime;
                double modR = r.norm();
                ThreeDVector nDotuDotT = r*( -3.0*(normalVector.dot(r)*delU.dot(r))/(4.0*M_PI*pow(modR, 5.0)) );
        
                res = res + nDotuDotT*(w[iQuad]*0.5*hS);
            }
        }
        return res;
    }

    //******Update area and x0********
    void updateArea()   {  area = integrateScalarfunc(&BIMobjects::getArea);  }

    void updateX0() { x0 = integrateVectorfunc(&BIMobjects::getX0)*(1.0/area);  }

    //*******define necessary functions for BIM*************
    static ThreeDVector getX0(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        //take x0 as zero here;
        return x;
    }
    
    static double getArea(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        return 1.0;
    }

    static ThreeDVector Drow1(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        ThreeDVector xBar = x-x0, res;
        double xBarSq = xBar.dot(xBar);
        res = ThreeDVector(1.0, 0.0, 0.0)*xBarSq - xBar*xBar.x[0];
        return res;
    }
    static ThreeDVector Drow2(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        ThreeDVector xBar = x-x0, res;
        double xBarSq = xBar.dot(xBar);
        res = ThreeDVector(0.0, 1.0, 0.0)*xBarSq - xBar*xBar.x[1];
        return res;
    }
    static ThreeDVector Drow3(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        ThreeDVector xBar = x-x0, res;
        double xBarSq = xBar.dot(xBar);
        res = ThreeDVector(0.0, 0.0, 1.0)*xBarSq - xBar*xBar.x[2];
        return res;
    }

    public:
    ThreeDVector uRB, omega;// set them after converging through the picard iterates!
    map<int, ThreeDVector> uS, uSNxt; // uS-> velocity for global index
    double area; 

    BIMobjects(ThreeDVector * pts, int size): mesh(pts, size) 
    {
        //set quadrature data:
        int nQ = 16;
        quadratureDat.open("./Headers/triangleQuadratures/p8.txt"); //FILE MUST BE PRESENT AT THIS LOCATION!
        if(!quadratureDat.is_open()) 
        {
            cout<<"Error reading file";
        }

        for (int iQuad = 0; iQuad < nQ; iQuad++)
        {
            double temp[3];
            quadratureDat>>temp[0];
            quadratureDat>>temp[1];
            quadratureDat>>temp[2];
            xi.push_back(temp[0]);
            eta.push_back(temp[1]);
            w.push_back(temp[2]);
        }
        quadratureDat.close();

        // update area and x0 once integration over mesh is defined:
       // updateArea();
    }

    void refineMesh(int nTimes)
    {
        mesh::refineMesh(nTimes);
        initializeUs();
        updateX0();
        updateArea();
        getItensor();
    }
    //*********define functions to transform the mesh*************
    // make ellipsoid out of sphere:
    void scale(double a, double b, double c)    // function overriden, use derived class to call functions of the base class.
    {
        mesh::scale(a, b, c);
        updateArea();
        updateX0();
        getItensor();
    }

    //rotate points by theta about nHat about x0:
    void rotate(ThreeDVector nHat, double theta) 
    {
        mesh::rotate(nHat, theta);
        getItensor();
    }

    void translate(ThreeDVector dx)
    {
        mesh::translate(dx);
    }

    //******* define functions for integration over elements ************
    // gives integral of f dot da, da pointing towards outer normal to the body
    //f = f(x, x0, uS, area, ItensorInv)
    double integratefuncDotDa(ThreeDVector (*func)(ThreeDVector, ThreeDVector, ThreeDVector, double, vector<ThreeDVector>)) 
    {
        double res = 0.0;
        for (int iElem = 0; iElem < element.size(); iElem++)
        {
                //get global coordinates of the element:
            ThreeDVector x1 = globalCoord[element[iElem].x[0]].x;
            ThreeDVector x2 = globalCoord[element[iElem].x[1]].x;
            ThreeDVector x3 = globalCoord[element[iElem].x[2]].x;
            // get midPoints of elements:
            ThreeDVector x4 = globalCoord[elementMid[iElem].x[0]].x;
            ThreeDVector x5 = globalCoord[elementMid[iElem].x[1]].x;
            ThreeDVector x6 = globalCoord[elementMid[iElem].x[2]].x;
            
            // map to standard triangle:
            // general vector inside the element: x = Sum_{i=1}^6 x_i phi_i
            
            for(int iQuad=0; iQuad < w.size(); iQuad++)
            {
                double xiIn = xi[iQuad], etaIn = eta[iQuad], zetaIn = 1.0 - xi[iQuad] - eta[iQuad];
                
                //basis vectors:
                ThreeDVector eXi = x1*(1.0-4.0*zetaIn) + x2*(4.0*xiIn - 1.0) + x4*(4.0*zetaIn - 4.0*xiIn) + x5*(4.0*etaIn) - x6*(4.0*etaIn);
                ThreeDVector eEta = x1*(1.0-4.0*zetaIn) + x3*(4.0*etaIn - 1.0) - x4*(4.0*xiIn) + x5*(4.0*xiIn) + x6*(4.0*zetaIn - 4.0*etaIn);
                
                double hS = eXi.cross(eEta).norm();
                ThreeDVector normalVector = eXi.cross(eEta)*(1.0/hS);
                
                ThreeDVector xElem = x1*(zetaIn*(2.0*zetaIn-1.0)) + x2*(xiIn*(2.0*xiIn - 1.0)) + x3*(etaIn*(2.0*etaIn-1.0)) + x4*(4.0*zetaIn*xiIn) + x5*(4.0*xiIn*etaIn) + x6*(4.0*etaIn*zetaIn);
                
                //linear interpolation for uS:
                ThreeDVector uSElem = uS[element[iElem].x[0]]*(zetaIn) + uS[element[iElem].x[1]]*(xiIn) + uS[element[iElem].x[2]]*(etaIn);

                res += w[iQuad]*func(xElem, x0, uSElem, area, ItensorInv).dot(normalVector)*0.5*hS;
            }
        }
        return res;
    }

    // pass relevant object's variables as argument to the static functions!
    //f = f(x, x0, uS, area, ItensorInv)
    ThreeDVector integrateVectorfunc(ThreeDVector (*func)(ThreeDVector, ThreeDVector, ThreeDVector, double, vector<ThreeDVector>))    
    {
        ThreeDVector res(0.0, 0.0, 0.0);
        for (int iElem = 0; iElem < element.size(); iElem++)
        {
            //get global coordinates of the element:
            ThreeDVector x1 = globalCoord[element[iElem].x[0]].x;
            ThreeDVector x2 = globalCoord[element[iElem].x[1]].x;
            ThreeDVector x3 = globalCoord[element[iElem].x[2]].x;
            // get midPoints of elements:
            ThreeDVector x4 = globalCoord[elementMid[iElem].x[0]].x;
            ThreeDVector x5 = globalCoord[elementMid[iElem].x[1]].x;
            ThreeDVector x6 = globalCoord[elementMid[iElem].x[2]].x;
            
            // map to standard triangle:
            // general vector inside the element: x = Sum_{i=1}^6 x_i phi_i
            
            for(int iQuad=0; iQuad < w.size(); iQuad++)
            {
                double xiIn = xi[iQuad], etaIn = eta[iQuad], zetaIn = 1.0 - xi[iQuad] - eta[iQuad];
                
                //basis vectors:
                ThreeDVector eXi = x1*(1.0-4.0*zetaIn) + x2*(4.0*xiIn - 1.0) + x4*(4.0*zetaIn - 4.0*xiIn) + x5*(4.0*etaIn) - x6*(4.0*etaIn);
                ThreeDVector eEta = x1*(1.0-4.0*zetaIn) + x3*(4.0*etaIn - 1.0) - x4*(4.0*xiIn) + x5*(4.0*xiIn) + x6*(4.0*zetaIn - 4.0*etaIn);
                
                double hS = eXi.cross(eEta).norm();
                
                ThreeDVector xElem = x1*(zetaIn*(2.0*zetaIn-1.0)) + x2*(xiIn*(2.0*xiIn - 1.0)) + x3*(etaIn*(2.0*etaIn-1.0)) + x4*(4.0*zetaIn*xiIn) + x5*(4.0*xiIn*etaIn) + x6*(4.0*etaIn*zetaIn);
                
                //linear interpolation for uS:
                ThreeDVector uSElem = uS[element[iElem].x[0]]*(zetaIn) + uS[element[iElem].x[1]]*(xiIn) + uS[element[iElem].x[2]]*(etaIn);

                res = res + func(xElem, x0, uSElem, area, ItensorInv)*(w[iQuad]*0.5*hS);
            }
        }
        return res;
    }

    // pass relevant object's variables as argument to the static functions!
    //f = f(x, x0, uS, area, ItensorInv)
    double integrateScalarfunc(double (*func)(ThreeDVector, ThreeDVector, ThreeDVector, double, vector<ThreeDVector>)) 
    {
        double res = 0.0;
        for (int iElem = 0; iElem < element.size(); iElem++)
        {
            //get global coordinates of the element:
            ThreeDVector x1 = globalCoord[element[iElem].x[0]].x;
            ThreeDVector x2 = globalCoord[element[iElem].x[1]].x;
            ThreeDVector x3 = globalCoord[element[iElem].x[2]].x;
            // get midPoints of elements:
            ThreeDVector x4 = globalCoord[elementMid[iElem].x[0]].x;
            ThreeDVector x5 = globalCoord[elementMid[iElem].x[1]].x;
            ThreeDVector x6 = globalCoord[elementMid[iElem].x[2]].x;
            
            // map to standard triangle:
            // general vector inside the element: x = Sum_{i=1}^6 x_i phi_i
            
            for(int iQuad=0; iQuad < w.size(); iQuad++)
            {
                double xiIn = xi[iQuad], etaIn = eta[iQuad], zetaIn = 1.0 - xi[iQuad] - eta[iQuad];
                
                //basis vectors:
                ThreeDVector eXi = x1*(1.0-4.0*zetaIn) + x2*(4.0*xiIn - 1.0) + x4*(4.0*zetaIn - 4.0*xiIn) + x5*(4.0*etaIn) - x6*(4.0*etaIn);
                ThreeDVector eEta = x1*(1.0-4.0*zetaIn) + x3*(4.0*etaIn - 1.0) - x4*(4.0*xiIn) + x5*(4.0*xiIn) + x6*(4.0*zetaIn - 4.0*etaIn);
                
                double hS = eXi.cross(eEta).norm();
                ThreeDVector normalVector = eXi.cross(eEta)*(1.0/hS);
                
                ThreeDVector xElem = x1*(zetaIn*(2.0*zetaIn-1.0)) + x2*(xiIn*(2.0*xiIn - 1.0)) + x3*(etaIn*(2.0*etaIn-1.0)) + x4*(4.0*zetaIn*xiIn) + x5*(4.0*xiIn*etaIn) + x6*(4.0*etaIn*zetaIn);
                
                //linear interpolation for uS:
                ThreeDVector uSElem = uS[element[iElem].x[0]]*(zetaIn) + uS[element[iElem].x[1]]*(xiIn) + uS[element[iElem].x[2]]*(etaIn);

                res = res + func(xElem, x0, uSElem, area, ItensorInv)*(w[iQuad]*0.5*hS);
            }
        }
        return res;
    }

    //integrate to get uRB.
    static ThreeDVector getURB(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        return uS*(1.0/area);   
    }
    //integrate to get omega.
    static ThreeDVector getOmegaRB(ThreeDVector x, ThreeDVector x0, ThreeDVector uS, double area, vector<ThreeDVector> ItensorInv)
    {
        ThreeDVector xBar = x-x0;
        ThreeDVector res = ThreeDVector(ItensorInv[0].dot( xBar.cross(uS) ), ItensorInv[1].dot( xBar.cross(uS) ), ItensorInv[2].dot( xBar.cross(uS) ));
        return res;
    }

    //********* BIE for ith element***************
    void picardIterate(int GIndx, int myStartGC, int myEndGC)
    {
        ThreeDVector res(0.0, 0.0, 0.0);

        if(GIndx<myStartGC || GIndx>=myEndGC)   
        {
            uSNxt[GIndx] = res;
        }

        else
        {
            ThreeDVector xPrime = globalCoord[GIndx];
            ThreeDVector Prb = uRBAux + omegaAux.cross(xPrime - x0);    
            ThreeDVector r = xPrime - x0, gHat(0.0, -1.0, 0.0);
            double modR = r.norm();

            //Add torque as well:
            ThreeDVector b = (gHat + r*(r.dot(gHat)/pow(modR, 2.0)) )*(3.0/(4.0*modR));    // size (a) = 1; sphere of radius a=1 falls with terminal speed = 1;
                
            for (int i = 0; i < elementsInGlobalIndx[GIndx].size(); i++)    // iterate over all elements sharing the common GIndx
            {
                int eIndx = elementsInGlobalIndx[GIndx][i]; // get the element index
                // Top is very expensive!!!
                ThreeDVector Top = integralDLOp(eIndx, GIndx);  
                res = res + b - Prb + uS[GIndx] - Top;    

            }
            uSNxt[GIndx] = res*(1.0/elementsInGlobalIndx[GIndx].size());    //take average of contribution from all elements sharing the GIndx.
            
        }
        
    }

    // call before picard iterate to set auxillary fields!
    void refreshuS()
    {  
        uS = uSNxt; 
        uRBAux = integrateVectorfunc(&BIMobjects::getURB);
        omegaAux = integrateVectorfunc(&BIMobjects::getOmegaRB);
    }

    // after picard iterations, get uS using uS->Prb[uS]; (earlier uS was auxillary!)
    //do all calculations/iterations with uS (auxillary) and finally replace uS by Prb[uS]
    void projectRB()
    {
        for (int iGC = 0; iGC < nCoordFlat; iGC++)
        {
            ThreeDVector xPrime = globalCoord[iGC];
            uS[iGC] = uRBAux + omegaAux.cross(xPrime-x0);
        }
    }

};