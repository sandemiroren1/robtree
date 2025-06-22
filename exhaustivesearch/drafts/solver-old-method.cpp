#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <atomic>
#include <thread>
#include <set>
#include <algorithm>
using namespace std;
#include "solver.h"
#include "sophisticated_struct.h"   
#include <iomanip>
#include <chrono>
#include <queue>


vector<char> misclassified;
vector<int> revert_stack;
vector<char> points_to_be_selected_left;
vector<char> points_to_be_selected_right;
vector<int> travel_order= {0,1,3,4,2,5,6};

vector<int> stack_for_reversions;
int timelimit = 1<<30;

chrono::seconds timelimit_s;
chrono::time_point<std::chrono::steady_clock> start;
int m;
int n;
int revert_stack_size = 0;
int misclassifiedCount = 0;
int nr_of_calls = 0;
vector<double> Xtransposed;
string output_filename;
double epsilon = 1e-7;
vector<double> epsilons_for_errors;
int warm_start_value = 1<<30;
vector<chrono::milliseconds> timings;
vector<int> upper_bounds;
bool record_progess;
int oldUpperBound = 0;
bool checkTime(){
    if ((chrono::steady_clock().now() - start) > timelimit_s) {
        return true;
    }
    return false;
}
queue<int> q;
int solve_depth1_robust_efficient(vector<char>& y,int s,vector<vector<double>>& data_points_sorted
    ,vector<char>& datapoints,int f0,vector<vector<int>>&indices_sorted,vector<double>& dR
    ,double a,double b,int f0Root){
    
    int bestValue = 1<<26;
    int curr_error = 0;
    int count_of_true = 0;
    int count_of_false = 0;
    for(int i = 0; i<n;++i){
        bool included = (Xtransposed[i+n*f0Root]>= a)&&(Xtransposed[i+n*f0Root]<=b);
        datapoints[i] = included;
        count_of_true+=(included&&y[i]);
        count_of_false+=(included&&!y[i]);

    }
    vector<int>& indices = indices_sorted[f0];
    if(count_of_false==0&&count_of_true==0){
        return 0;
    }
    for(int j = 0; j<4;++j){
        bool leftClassification = j == 0 ||j ==2;
        bool rightClassification = j == 1 || j ==2;
        curr_error = (1-rightClassification)*count_of_false+(rightClassification*count_of_true);
        int curr_pointer = 0;
        for(int i = 0 ; i < data_points_sorted[f0].size();++i){
            
            double threshold = data_points_sorted[f0][i]+ epsilons_for_errors[f0]+dR[f0];
            
            while(q.size()>0&&Xtransposed[q.front()+f0*n]+dR[f0]<=threshold){
                int indice = q.front();
                q.pop();
                bool curr_classification = y[indice];
                curr_error-=rightClassification!=curr_classification&&leftClassification==curr_classification;

            }
            while(curr_pointer<n&&Xtransposed[indices[curr_pointer]+n*f0]-dR[f0]<=threshold){
                int the_indice = indices[curr_pointer];
                curr_pointer++;
                if(!datapoints[the_indice]){
                    continue;
                }
                bool current_classification = y[the_indice];
                if(Xtransposed[the_indice+n*f0]+dR[f0]<=threshold){
                    curr_error-=current_classification==leftClassification&&current_classification!=rightClassification;
                    curr_error+=current_classification!=leftClassification&&current_classification==rightClassification;
                    continue;
                }
                if(rightClassification==current_classification){
                    if(leftClassification!=current_classification){
                        curr_error++;
                        q.push(the_indice);
                    }
                }else{
                    if(leftClassification==current_classification){
                        q.push(the_indice);
                    }
                }
            }
            bestValue=min(bestValue,curr_error);
        }
    }

    
    return bestValue;

}
int solve_depth1_robust(vector<double>& X,vector<char>& y,int s,vector<vector<double>>& data_points_sorted
    ,vector<char>& datapoints,int f0,vector<vector<int>>&indices_sorted,vector<double>& dR
    ,double a,double b,int f0Root){
    
    int bestValue = 1<<26;
    int currError = 0;
    bool cnt = false;
    for(int i = 0; i<n;++i){
        datapoints[i] = (X[i*m+f0Root]> a)&&(X[i*m+f0Root]<b);
    }

    for(int j = 0; j<4;++j){
        bool leftClassification = j == 0 ||j ==2;
        bool rightClassification = j == 1 || j ==2;
        for(int i = 0 ; i < n;++i){
            if(!datapoints[i]){
                continue;
            }
            cnt = true;
            int curr_error = 0;
            double threshold = X[i*m+f0]+ epsilons_for_errors[f0]+dR[f0];
            for(int k = 0; k < n;++k){
                if(!datapoints[k]){
                    continue;
                }
                double data_value = X[k*m+f0];
                bool data_classification = y[k];
                bool misclassified_left = (data_value-dR[f0]<=threshold&&(data_classification!=leftClassification));
                bool misclassified_right = (data_value+dR[f0]>threshold&&(data_classification!=rightClassification));
                curr_error+=(misclassified_left||misclassified_right);
            }
            bestValue=min(bestValue,curr_error);

        }
    }
    if(!cnt){
        return 0;
    }

    return bestValue;

}
int solve_depth1(vector<double>& X,vector<char>& y,int s,vector<vector<double>>& data_points_sorted
    ,vector<char>& datapoints,int f0,vector<vector<int>>&indices_sorted,vector<double>& dR
    ,double a,double b,int f0Root){
    
    int bestValue = 1<<26;
    int currError = 0;
    int currPointer = 0;
    int ctr=0;
    int ctl = 0;
    int cfr = 0;
    int cfl = 0;
    for(int i = 0; i<n;++i){
        datapoints[i] = (X[i*m+f0Root]>= a)&&(X[i*m+f0Root]<=b);
    }

    for(int i = 0 ; i<n;++i){
        cfr += (!y[i])&&datapoints[i];
        ctr += y[i]&&datapoints[i];
    }
    
    for(int i = 0; i<data_points_sorted[f0].size();++i){

        double threshold = data_points_sorted[f0][i] + dR[f0]+epsilons_for_errors[f0];
        while(currPointer < n && X[indices_sorted[f0][currPointer]*m+f0] <= threshold){
            if(datapoints[indices_sorted[f0][currPointer]]){
                if(y[indices_sorted[f0][currPointer]]){
                    ctl++;
                    ctr--;
                }else{
                    cfl++;
                    cfr--;
                }
            }
            currPointer++;
        }
        currError = 0;
       
        if (ctl > cfl){
            currError += cfl;
        }else{
            currError += ctl;
        }
        if(ctr > cfr){
            currError += cfr;
        }else{
            currError += ctr;
        }
        if(currError < bestValue){
            bestValue = currError;
        }
        
    }


    return bestValue;

}

int handle_new_misclassifications(vector<vector<double>>& data_points_sorted,vector<Variable2>& variables, 
        bool chosen_class,vector<char>& Y,vector<double>& dL,vector<double>& dR
        ,int index,int upperbound,vector<vector<int>>& indices_sorted){
    int newWrongs = 0;
    int parentIndex = (index-1)>>1;
    int parent_parentIndex = (parentIndex-1)>>1;

    const Variable2& parentVariable = variables[parentIndex];
    const Variable2& parentParentVariable = variables[parent_parentIndex];

    const bool isLeftOfParent = (((parentIndex<<1)+1) == index);
    const bool isLeftOfParentParent = (((parent_parentIndex<<1)+1) == parentIndex);

    const double dRParent = dR[parentVariable.chosenDim];
    const double dLParent = dL[parentVariable.chosenDim];
    const double dRParentParent = dR[parentParentVariable.chosenDim];
    const double dLParentParent = dL[parentParentVariable.chosenDim];

    const double parentSplit_plus_DL = parentVariable.chosenSplit+dLParent;
    const double parentSplit_minus_DR = parentVariable.chosenSplit-dRParent;

    const double parentParentSplit_plus_DL = parentParentVariable.chosenSplit+dLParentParent;
    const double parentParentSplit_minus_DR = parentParentVariable.chosenSplit-dRParentParent;


    const int parentDim = parentVariable.chosenDim;
    const int parentParentDim = parentParentVariable.chosenDim;
    const int accessing_int_parent_dim = n*parentDim;
    const int accessing_int_parent_parent_dim = n*parentParentDim;

    const vector<int>& sorted_indices = indices_sorted[parentDim];
    if (isLeftOfParent) [[likely]]{
        for(int j = 0; j<n;++j){

            int i = sorted_indices[j];
            
            bool continue_with_work = !((Y[i] == chosen_class) | misclassified[i]);
            
            double theValue = Xtransposed[accessing_int_parent_dim+i];
            double theValue2 = Xtransposed[accessing_int_parent_parent_dim+i];

            bool is_data_included_p1 = (theValue <= parentSplit_plus_DL); 
            j = ((!is_data_included_p1)<<20)+j;              

            bool flag = continue_with_work&is_data_included_p1&((isLeftOfParentParent & (theValue2  <= parentParentSplit_plus_DL)) 
                            || (!isLeftOfParentParent & (theValue2  > parentParentSplit_minus_DR)));
            
            
            
            misclassifiedCount+=(flag);
            newWrongs+=(flag);
            revert_stack[revert_stack_size] = i;
            revert_stack_size+=(flag);
            misclassified[i]|= (flag);
            
            
            
        }
    }else [[unlikely]]{
        for(int j = n-1; j>=0;j--){
        
            int i = indices_sorted[parentDim][j];

            bool continue_with_work = !((Y[i] == chosen_class) | (misclassified[i]));

            double theValue = Xtransposed[accessing_int_parent_dim+i];
            double theValue2 = Xtransposed[accessing_int_parent_parent_dim+i];

            bool is_data_included_p1 = (theValue  > parentSplit_minus_DR);
            j = j-((!is_data_included_p1)<<20);              
            
            bool flag = continue_with_work&is_data_included_p1&((isLeftOfParentParent & (theValue2 <= parentParentSplit_plus_DL)) 
                            || (!isLeftOfParentParent & (theValue2  > parentParentSplit_minus_DR)));
            
            
            
            misclassifiedCount+=(flag);
            newWrongs+=(flag);
            revert_stack[revert_stack_size] = i;
            revert_stack_size+=(flag);
            misclassified[i]|= (flag);
            
        }
    }

    return newWrongs;
}
void revert_misclassifications(int newWrongs){
    while(newWrongs > 0){
        int e = revert_stack[--revert_stack_size];
        --newWrongs;
        misclassified[e] = false;
        misclassifiedCount--;

    }
}
int backtracking(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,vector<Variable2>& variables,int index_in_travel,T2& tree,vector<vector<int>>& indices_sorted){
        nr_of_calls++;
        const int upperbound_error = upperbound->error;
        if (checkTime()) {
            return 1<<30;
        }
        if(misclassifiedCount>=upperbound->error){
            return misclassifiedCount;
        }
        int resValue = 1<<30;

        if (variables.size()==index_in_travel){
                
            if (misclassifiedCount < upperbound_error){
                upperbound->error = misclassifiedCount;
                for(int i = 0 ; i<variables.size();++i){
                    upperbound->tree[i].chosenDim=variables[i].chosenDim;
                    upperbound->tree[i].chosenSplit=variables[i].chosenSplit;
                    upperbound->tree[i].isLeaf=variables[i].isLeaf;
                    upperbound->tree[i].classification=variables[i].classification;
                }
            }
            
            return misclassifiedCount;
        }
        int index = travel_order[index_in_travel];

        Variable2& currentVariable =(variables[index]);
        if(index == 0){
            int skipRemaining = 0;
            variables[0].chosenDim = tree.f0;
            int a = tree.a;
            int b = tree.b;
            int dim = tree.f0;
            int startPoint = 0;
            double startValue = data_points_sorted[dim][tree.a];
            while(startPoint < n&&abs(X[m*indices_sorted[dim][startPoint]+dim]- startValue) > epsilon){
                startPoint++;
            }
            int endPoint = n-1;
            double endValue = data_points_sorted[dim][tree.b];
            while(endPoint>=0 && abs(X[m*indices_sorted[dim][endPoint]+dim]- endValue) > epsilon ){
                endPoint--;
            }
            double prev = 1<<30;
            for(int i = startPoint; i<=endPoint;++i){
                if(skipRemaining > 0){
                    skipRemaining--;
                    continue;
                }
                variables[0].chosenSplit = X[m*indices_sorted[dim][i]+dim]+dR[tree.f0]+epsilons_for_errors[dim];
                
                if (abs(variables[0].chosenSplit - prev) > epsilon){
                    prev = variables[0].chosenSplit;
                    int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
                    
                    skipRemaining = backTrack - upperbound->error;
                    if(backTrack < resValue){
                        resValue = backTrack;
                    }
                }
            }
            
            return resValue;
        }
        if (index == 4||index==6){
            //cout<<"LEAF\n";
            currentVariable.classification = !variables[index-1].classification;
            int newWrongs = handle_new_misclassifications(data_points_sorted,variables,currentVariable.classification,y,dL,dR,index,upperbound_error,indices_sorted);
            resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
            revert_misclassifications(newWrongs);
            return resValue;
        }
        if (index == 3||index==5){
            //cout<<"LEAF\n";
            currentVariable.classification = false;
            int newWrongs = handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,index,upperbound_error,indices_sorted);
            resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
            revert_misclassifications(newWrongs);
            currentVariable.classification = true;
            newWrongs = handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,index,upperbound->error,indices_sorted);
            int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
            revert_misclassifications(newWrongs);

            if(backTrack < resValue){
                resValue = backTrack;
            }
            return resValue;
        }
        vector<int> dims = tree.F1;
        if (index == 2){
            dims = tree.F2;
        }
        bool isLeft = (index == 1);
        double prev = 0;
        double finger = 0;
        int currI = 0;
        int skipRemaining = 0;

        variables[2*index+1].classification=false;
        variables[2*index+2].classification=false;
        currentVariable.chosenDim=0;
        currentVariable.chosenSplit=0;
        int newWrongs =  handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,2*index+1,upperbound_error,indices_sorted);
        newWrongs +=     handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,2*index+2,upperbound_error,indices_sorted);
        resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+3,tree,indices_sorted);
        revert_misclassifications(newWrongs);
        variables[2*index+1].classification=true;
        variables[2*index+2].classification=true;
        newWrongs =   handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,2*index+1,upperbound_error,indices_sorted);
        newWrongs +=  handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,2*index+2,upperbound_error,indices_sorted);
        resValue = min(resValue,backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+3,tree,indices_sorted));
        revert_misclassifications(newWrongs);

        for(int dim : dims){
            currentVariable.chosenDim = dim;
            prev = 1<<30;    
        
            for(int i = 0 ; i < n; i++){
                currI = indices_sorted[dim][i];
                bool valid_split = isLeft&&X[currI*m+variables[0].chosenDim] - dL[variables[0].chosenDim] <= variables[0].chosenSplit;
                valid_split = valid_split || (!isLeft && X[currI*m+variables[0].chosenDim] + dR[variables[0].chosenDim] > variables[0].chosenSplit);
                finger = X[currI*m+dim];
                if(!valid_split){
                    continue;
                }
                skipRemaining--;
                if((abs(finger - prev )> epsilon)&&skipRemaining < 0){

                    currentVariable.chosenSplit = finger+dR[dim]+epsilons_for_errors[dim];
                    int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
                    skipRemaining = backTrack - upperbound->error;
                    if (skipRemaining < 0){
                        skipRemaining = 0;
                    }
                    prev = finger;
                    if(backTrack < resValue){
                        resValue = backTrack;
                    }
                }
                
            }

        }
        return resValue;

}


void exhaustive_search_depth_2(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,T2& tree,vector<vector<int>>& indices_sorted,vector<Variable2>& variables){
    misclassifiedCount = 0;
    revert_stack_size = 0;

    backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,0,tree,indices_sorted);

}

vector<T2> prepareAL0(int m,vector<vector<double>>& data_points_sorted){
    vector<T2> AL2;
    for(int i = 0; i<m;++i){
        vector<int> F1;
        vector<int> F2;
        int size = data_points_sorted[i].size();
        for(int j = 0; j<m;++j){
            F1.push_back(j);
            F2.push_back(j);
        }
        sort(F1.begin(),F1.end(),[&](int i,int j){return data_points_sorted[i].size()<data_points_sorted[j].size();});
        sort(F2.begin(),F2.end(),[&](int i,int j){return data_points_sorted[i].size()<data_points_sorted[j].size();});
        AL2.push_back(T2(i,0,size-1,F1,F2));
    }
    sort(AL2.begin(),AL2.end(),[&](T2 i,T2 j){return data_points_sorted[i.f0].size()<data_points_sorted[j.f0].size();});
    
    return AL2;
}
void QuantBNB2(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,vector<vector<int>>& indices_sorted){
    vector<T2> ALk = prepareAL0(m,data_points_sorted);
    int current_size = ALk.size();
    vector<Variable2> sol(7);
    vector<Variable2> fakeSol(7);
    vector<Variable2> currSol(7);
    stack_for_reversions= vector<int>(n+10,-1);
    for(int i = 0; i<7;++i){
        bool isLeaf = i>2;
        sol[i] = Variable2(isLeaf);
        currSol[i] = Variable2(isLeaf);
    }
    UB* upper_bound = new UB(sol,warm_start_value);
    revert_stack = vector<int>(n,-1);
    misclassified = vector<char>(n+1,false);
    points_to_be_selected_right = vector<char>(n,false);

    vector<int> oneElementF1(1,0);
    vector<int> oneElementF2(1,0);
    vector<int> F2Best(m,1<<30);
    T2 curr = T2(-1,-1,-1,oneElementF1,oneElementF2);
    int alkSize = ALk.size();
    while (true){
        
        if(record_progess && oldUpperBound!=upper_bound->error){
                oldUpperBound=upper_bound->error;
                auto e = (chrono::steady_clock().now() - start);
                timings.push_back(chrono::duration_cast<chrono::milliseconds>(e));
                upper_bounds.push_back(upper_bound->error);
        }
        if (checkTime()) {
            goto end;
        }
        if(alkSize==0){
            goto end;
        }
        vector<T2> ALkpp;
        int newAlkSize = 0;
        for(int o = 0; o<alkSize;++o){
            alkSize = 0;
            T2 al = ALk.back();
            ALk.pop_back();            
            int f0 = al.f0;
            

            int a = al.a;
            int b = al.b;
            vector<int> F1 = al.F1;
            vector<int> F2 = al.F2;
            if (b-a<=s){
                // brute forcing
                exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,al,indices_sorted,currSol);
            }else{
                    // compute UB
                    for(int f1: F1){
                        for(int f2: F2){
                            // let t_0,...,t_s be almost-s equi spaced
                            for(int j = 0; j<=s; j++){
                                int tj = (a+(int)((j*1.0*(b-a)))/(1.0*s));
                                if(tj>b){
                                    break;
                                }
                                curr.f0 = f0;
                                curr.a = tj;
                                curr.b = tj;
                                curr.F1[0] = f1;
                                curr.F2[0] = f2;
                                exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,curr,indices_sorted,currSol);
                            }
                        }
                    }                  
                    // let t_0,...,t_s be almost-s equi spaced in [a,b]
                    double lower_uf = 3;
                    for(int j = 1; j<=s; j++){
                        vector<int> F1j= vector<int>();
                        vector<int> F2j= vector<int>();
                        int tj = (a+(int)((j*1.0*(b-a)))/(1.0*s));
                        if(tj>b){
                            break;
                        }
                        int tjmm = (a+(int)(((j-1)*1.0*(b-a)))/(1.0*s));
                        double lower_a = data_points_sorted[f0][tjmm]-epsilons_for_errors[f0];
                        //double lower_a_pp = lower_a+2*epsilons_for_errors[f0];
                        double lower_b = data_points_sorted[f0][tj]+ epsilons_for_errors[f0]; 
                        //double lower_b_mm = lower_b- 2*epsilons_for_errors[f0]; 

                        for(int i = 0 ; i<m;++i){
                           F2Best[i]= 1<<25;
                        }
                    
                        for(int f1: F1){
                            int maxError = 1<<26;
                            for(int f2: F2){
                                if (checkTime()) {
                                    goto end;
                                }
                                  
                                int l = solve_depth1_robust_efficient(y,s,data_points_sorted,points_to_be_selected_right,f1,indices_sorted,dR,-1,lower_a,f0);
                                int r = solve_depth1_robust_efficient(y,s,data_points_sorted,points_to_be_selected_right,f2,indices_sorted,dR,lower_b,lower_uf,f0);
                                int m =1<<26;
                                int s_prime = floor((0.6*n*s)/(1.0*(tjmm)-(tj)));
                                /*
                                0 0.3841671
                                2 0.3778261
                                2 0.2256521
                                */
                                // let t_0,...,t_s' be almost-s' equi spaced in [t_{j-1}+1,tj-1]
                                int new_a = tjmm;
                                int new_b = tj;
                                if(tj-tjmm<=s_prime){
                                    for(int j_prime = 1; j_prime<=s_prime; j_prime++){
                                        int tj_prime = (new_a+(int)((j_prime*1.0*(new_b-new_a)))/(1.0*s_prime));
                                        int tjmm_prime = (new_a+(int)(((j_prime-1)*1.0*(new_b-new_a)))/(1.0*s_prime));
                                        double b_prime = data_points_sorted[f0][tj_prime]+epsilons_for_errors[f0];
                                        double a_prime = data_points_sorted[f0][tjmm_prime]-epsilons_for_errors[f0];
                                        int m_left = solve_depth1_robust_efficient(y,s,data_points_sorted,points_to_be_selected_right,f1,indices_sorted,dR,lower_a,a_prime,f0);
                                        int m_right = solve_depth1_robust_efficient(y,s,data_points_sorted,points_to_be_selected_right,f2,indices_sorted,dR,b_prime,lower_b,f0);
                                        m = min(m,m_left+m_right);
                                    }
                                }else{
                                    m=0;
                                }
                                
                                if(l+r+m<maxError){
                                    maxError = l+r+m;
                                }
                                if(l+r+m<F2Best[f2]){
                                    F2Best[f2]=l+r+m;
                                }      
                                                     
                            }
                
                            if(maxError<upper_bound->error){
                                F1j.push_back(f1);
                            }
                        }
                        for(int i = 0; i<m;++i){
                            if(F2Best[i]<upper_bound->error){
                                F2j.push_back(i);
                            }
                        }
                         
                        if(F1j.size()==0||F2j.size()==0){
                            continue;
                        }else{
                            ALkpp.push_back(T2(f0,tjmm,tj-1,F1j,F2j));                   
                        }
                    }                
            }
        }
        for(T2 e : ALkpp){
            ALk.push_back(e);
        }
        alkSize = ALk.size();
    }
 end:
    
    //outfile<<setprecision(10);
    //outfile<<"2";
    cout<<"2";
    for(auto& e : upper_bound->tree){
        //outfile<<"\n"<<e;
        cout<<"\n"<<e;
    }
    cout<<"\n";
    for(int i = 0 ; i < upper_bounds.size();++i){
        cout << timings[i].count()<<" "<<upper_bounds[i]<<"\n";
    }
    cout<<"***\n";
    //outfile<<"\nerror "<<upper_bound->error;
    cout<<"\nerror "<<upper_bound->error;
    cout<<"\nnumber of calls: "<<nr_of_calls<<"\n";
    if(!checkTime()){
        cout<<"True";
    }else{
        cout<<"False";
    }
    


}


int main(int argc, char* argv0[]) {
    // First argument is the filename
    string filename = argv0[1];
    cout<<setprecision(7);

    /*
    argv0 structure:
    [0]: n, the number of data
    [1] : m, the number of features
    [2,1+m] : delta l
    [2+m,1+2m] : delta r
    [2+2m] : timelimit
    [3+2m] : depth
    [4+2m] : s
    [4+2m+1,4+2m + m*n] : The data, segments of m points represent each data point.
    [5+2m+m*n, 5+2m+m*n+n] : The classification of each data point

    */
   ifstream myfile (filename);
   vector<string> argv;
   if( myfile.is_open()){
        //cout<<"File opened successfully\n";
        while(myfile.good()){
            string line;
            getline(myfile,line);
            //cout<<"read line: "<<line<<endl;
            argv.push_back(line);
        }
   }
   


    int index = 0;
    n = stoi(argv[index++]);
    m = stoi(argv[index++]);
    int misclassifiedCount = 0;
    vector<char> misclassified = vector<char>(n+10,false);
    vector<double> deltaL = vector<double>(m,0);
    vector<double> deltaR = vector<double>(m,0);
    for(int i = 0; i<m;++i){
        deltaL[i] = stod(argv[index++]);
    }
    for(int i = 0; i<m;++i){
        deltaR[i] = stod(argv[index++]);
    }
    
    timelimit = stoi(argv[index++]);
    int depth = stoi(argv[index++]);
    int s = stoi(argv[index++]);
    epsilon = stold(argv[index++]);
    warm_start_value = stoi(argv[index++]);
    record_progess = stoi(argv[index++]);
    vector<double> X = vector<double>(n*m,0);
    Xtransposed = vector<double>(m*n);
    vector<char> Y = vector<char>(n);
    for(int i = 0; i<n;++i){
        for(int j = 0; j<m;++j){
            X[i*m+j] = stod(argv[index++]);
            Xtransposed[j*n+i] = X[i*m+j];
        }
    }
    for(int i =0; i<n;++i){
        Y[i] = stoi(argv[index++]);
    }
    // matrix of data points in which each row is the unique data points for that feature sorted ascendingly

    vector<vector<double>> data_points_sorted = vector<vector<double>>(m);
    for(int i = 0; i<m;++i){
        vector<double> sorted;
        for(int j = 0; j<n;++j){
            sorted.push_back(X[j*m+i]);
        }
        sort(sorted.begin(),sorted.end());
        vector<double> result;
        for (size_t j = 1; j < sorted.size(); ++j) {
            if (result.size()==0||abs(sorted[j] - result.back()) > 1e-7) {
                result.push_back(sorted[j]);
            }
        }
        data_points_sorted[i] =result;
    }
    vector<vector<int>> indices_sorted(m,vector<int>(n,-1));
    for(int i= 0; i <m;++i){
        for(int j = 0; j<n;++j){
            indices_sorted[i][j] = j;
            
        }
        sort(indices_sorted[i].begin(), indices_sorted[i].end(), [&](int a, int b) {
            return X[a*m+i] < X[b*m+i];
        });
        
    }
    start = std::chrono::steady_clock::now();
    timelimit_s = std::chrono::seconds(timelimit);
    epsilons_for_errors=vector<double>(m);
    for(int i = 0; i <m;++i){
        double e_choice = 1<<30;
        if(data_points_sorted[i].size()==1){
            e_choice = 0;
            epsilons_for_errors[i] = e_choice;

            continue;
        }
        for(int j = 0; j < data_points_sorted[i].size()-1;++j){
            e_choice = min(e_choice,data_points_sorted[i][j+1]-data_points_sorted[i][j]);
        }
        epsilons_for_errors[i] = epsilon;
    }
    QuantBNB2(X,Y,deltaL,deltaR,s,data_points_sorted,indices_sorted);
   
    
    

    return 0;
}
