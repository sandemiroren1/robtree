#include <vector>
using namespace std;




struct T1 {
    int f0;
    int a;
    int b;
    T1(int f0, int a, int b) : f0(f0),a(a),b(b) {}
};
struct T2 {
    //T2 = (f0, [a,b],F1,F2)
    int f0;
    int a;
    int b;
    vector<int> F1;
    vector<int> F2;
    T2(int f0, int a, int b, vector<int> F1, vector<int> F2) : f0(f0), a(a), b(b), F1(F1), F2(F2) {}


    friend std::ostream& operator<<(std::ostream& os, const T2& var) {
        os << "f0: " << var.f0 << " a: " << var.a << " b: " << var.b << " F1: { ";
        for (const auto& f : var.F1) {
            os << f << " ";
        }
        os<<"}\n";
        os << "F2: {";
        for (const auto& f : var.F2) {
            os << f << " ";
        }
        os << "}\n";
        return os;
    }
};



struct Variable2{
    float chosenSplit;
    int chosenDim;
    bool isLeaf;
    bool classification;
    Variable2(){
        this->isLeaf = false;
        this->chosenSplit = -1;
        this->chosenDim = -1;
        this->classification = false;
    }
    Variable2(bool isLeaf){
        this->isLeaf = isLeaf;
        this->chosenSplit = -1;
        this->chosenDim = -1;
    }
    // Friend function to enable std::cout << variable
    friend std::ostream& operator<<(std::ostream& os, const Variable2& var) {
        if(var.isLeaf){
            os << "c"<<var.classification;
        }else{
            os << var.chosenDim << " " << var.chosenSplit;
        }
        return os;
    }

};
struct ResultNode {

};
struct Leaf : ResultNode {
    bool classification;
};
struct UB{
    vector<Variable2>& tree;
    int error;

    int current_working_theta_L;
    int current_working_theta_R;

    int current_working_theta_LL;
    int current_working_theta_RR;
    UB(vector<Variable2>& tree,int error) : tree(tree),error(error) {}
};

struct DoubleComparator {
    bool operator()(double a, double b) const {
        // Return true if a is less than b, considering EPSILON
        return (b - a) > 1e-7;
    }
};