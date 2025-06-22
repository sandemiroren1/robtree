#include <vector>
#include <ostream>
using namespace std;
#include <iostream>
#include <cstdlib>



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
    double chosenSplit;
    int chosenDim;
    char isLeaf;
    char classification;
    vector<char> datapoints;
    Variable2(){
        this->isLeaf = false;
        this->chosenSplit = -1;
        this->chosenDim = -1;
        this->classification = false;
        this->datapoints = vector<char>();

    }
    Variable2(char isLeaf){
        this->isLeaf = isLeaf;
        this->chosenSplit = -1;
        this->chosenDim = -1;
        this->datapoints = vector<char>();
    }
    // Friend function to enable std::cout << variable
    friend std::ostream& operator<<(std::ostream& os, const Variable2& var) {
        if(var.isLeaf){
            os << "c"<<(int)var.classification;
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
    UB(vector<Variable2>& tree,int error) : tree(tree),error(error) {}
};

struct DoubleComparator {
    bool operator()(double a, double b) const {
        // Return true if a is less than b, considering EPSILON
        return (b - a) > 1e-7;
    }
};


 
// Define the default capacity of a queue
#define SIZE 1000
 
// A class to store a queue
class Queue
{
 // https://cs.kenyon.edu/index.php/scmp-218-00-data-structures/queue-implementation-in-c/
public:
    Queue(int size = SIZE);     // constructor
    ~Queue();                   // destructor
    vector<int> arr;       // array to store queue elements
    int capacity;   // maximum capacity of the queue
    int front;      // front points to the front element in the queue (if any)
    int rear;       // rear points to the last element in the queue
    int count;      // current size of the queue
    int dequeue();
    void enqueue(int x);
    int peek();
    int size();
    bool isEmpty();
    bool isFull();
};
 
// Constructor to initialize a queue
Queue::Queue(int size)
{
    arr = vector<int>(size,0);
    capacity = size;
    front = 0;
    rear = -1;
    count = 0;
}
 
// Destructor to free memory allocated to the queue
Queue::~Queue() {
    
}
 
// Utility function to dequeue the front element
int Queue::dequeue()
{

    int x = arr[front]; 
    front = (front + 1) % capacity;
    count--;
 
    return x;
}
 
// Utility function to add an item to the queue
void Queue::enqueue(int item)
{
    rear = (rear + 1) % capacity;
    arr[rear] = item;
    count++;
}
 
// Utility function to return the front element of the queue
int Queue::peek()
{
    return arr[front];
}
 
// Utility function to return the size of the queue
int Queue::size() {
    return count;
}
 
// Utility function to check if the queue is empty or not
bool Queue::isEmpty() {
    return (size() == 0);
}
 
// Utility function to check if the queue is full or not
bool Queue::isFull() {
    return (size() == capacity);
}
 