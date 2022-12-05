#include<math.h>

class dumbbellParams
{   
    private:
    double a, d;

    double fn(double c)
    {
        return pow(a*a-c*c, 3.0) - a*(a*a+3*c*c)*pow(c*c+d*d, 1.5);
    }
    double Dfn(double c)
    {
        double dx = 0.0001;
        return (fn(c+dx) - fn(c-dx))/(2.0*dx);
    }
    
    double getC()
    {
        double currC = a;
        double nextC;
        for (int i = 0; i < 20; i++)
        {
            if(Dfn(currC) == 0.0 || currC<0.0 || currC > a) break;
            
            nextC = currC - fn(currC)/Dfn(currC);
            
            if(pow(nextC-currC, 2.0) < pow(10.0, -12.0))    break;
            
            currC = nextC;
        }
        
        return nextC;
    }

    double eqn(double r, double cTheta)
    {
        return pow( r*r+c*c+2.0*r*c*cTheta , -1.5) + pow(r*r+c*c-2.0*c*r*cTheta, -1.5) - 2.0*pow(c*c + d*d, -1.5);
    }

    double Deqn(double r, double cTheta)
    {
        double dx = 0.0001;
        return (eqn(r+dx, cTheta)-eqn(r-dx, cTheta))/(2.0*dx);
    }   

    public:
    double c;
    dumbbellParams(double a, double d)
    {
        this->a = a;
        this->d = d;
        c = getC();
    }
    
    double getR(double cTheta) //R is radial coordinate in spherical coord.
    {
        double currR = d, fCurr;
        double nextR = a, fNext, middle;
        for (int i = 0; i < 50; i++)
        {
            fCurr =  eqn(d, cTheta);
            fNext = eqn(a, cTheta);
            double m = (fNext-fCurr)/(nextR - currR);            
            
            middle = nextR - fNext/m;
            
            if(eqn(middle, cTheta)*fCurr >=0.0) currR = middle;
            else    nextR = middle;
            
            if(pow(eqn(middle, cTheta), 2.0) < pow(10.0, -10.0)) break;

        }
        return middle;
    } 

};