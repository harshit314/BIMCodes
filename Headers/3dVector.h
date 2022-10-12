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
