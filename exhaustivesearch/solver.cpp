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
#include <iomanip>
#include <chrono>
#include <queue>
#include <climits>
/*
    * Exhaustive search solver for the robust decision tree problem.
    * This solver is designed to find the optimal decision tree structure by exploring all possible configurations.
    * Uses Quant-BnB and ConTree ideas.
*/
vector<int> backup_index_1;
vector<int> backup_index_2;
int curr_error = 0;
const int infty = 1<<26;
vector<int> misclassified_per_datapoint;
vector<char> points_to_be_selected_right;
vector<int> travel_order= {0,1,2,3,4,5,6};
Queue clusterCs[3];
int timelimit = 1<<30;
chrono::seconds timelimit_s;
chrono::time_point<std::chrono::steady_clock> start;
int m;
int n;
int revert_stack_size = 0;
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
int prunedF1 = 0;
int prunedF2 = 0;
Queue q;

/// @brief Checks if the time limit has been exceeded
/// @return true if the time limit has been exceeded, false otherwise
bool checkTime(){
    if ((chrono::steady_clock().now() - start) > timelimit_s) {
        return true;
    }
    return false;
}


/// @brief Specialized solver for depth 1 trees given data points for the lower bounds. 
/// @param y The labels of the data points.
/// @param data_points_sorted The sorted data points.
/// @param datapoints Auxillary array to reuse memory.
/// @param f0 The current feature.
/// @param indices_sorted The sorted indices of the data points per feature.
/// @param dR Right perturbations per feature.
/// @param dL Left perturbations per feature.
/// @param a The left threshold. 
/// @param b The right threshold. It operates on values in $D^{f_0}_{[a,b]}$
/// @param f0Root The root feature.
/// @param X The feature matrix.
/// @return the minimum number of misclassified points
int solve_depth1_robust_efficient(vector<char>& y,vector<vector<double>>& data_points_sorted
    ,vector<char>& datapoints,int f0,vector<vector<int>>&indices_sorted,vector<double>& dR,vector<double>& dL
    ,double a,double b,int f0Root,vector<double>& X){
    
    int bestValue = infty;
    int curr_error_here = 0;
    int count_of_true = 0;
    int count_of_false = 0;

    // Decide which data points are going to be considered.
    for(int i = 0; i<n;++i){
        bool included = (X[i*m+f0Root]>= a)&&(X[i*m+f0Root]<=b);
        datapoints[i] = included;
        count_of_true+=(included&&y[i]);
        count_of_false+=(included&&!y[i]);
    }
    
    // Equivallent to having both leafs classifying true/false
    bestValue = min(count_of_true,count_of_false);
    for(int j = 0; j<2;++j){
        // iterate through the other leaf combinations
        bool leftClassification = j == 0 ;
        bool rightClassification = j == 1;

        // points are ininitially directed to the right subtree
        if(rightClassification){
            curr_error_here = count_of_false;
        }else{
            curr_error_here = count_of_true;
        }

        // reset current pointer and the queue
        int curr_pointer = 0;
        q.count = 0;
        q.front = 0;
        q.rear = -1;
        double prev = -5;
        // iterate over all thresholds
        for(int p = 0 ; p < n;++p){
            int i = indices_sorted[f0][p];
            if(!datapoints[i]){
                continue;
            }
            double threshold = X[i*m+f0]+ epsilons_for_errors[f0]+dR[f0];
            
            if(abs(prev-threshold)<epsilons_for_errors[f0]){
                continue;
            }
//            cout<<" Error before: "<<curr_error_here<<" threshold "<<threshold<<"\n";

            prev = threshold;

            // Check if there are any points that were on boths sides that are now only on the left
            while(q.size()>0&&X[m*q.peek()+f0]+dR[f0]<=threshold){
                int indice = q.peek();
                q.dequeue();
                bool curr_classification = y[indice];
                if(rightClassification!=curr_classification&& leftClassification==curr_classification){
                    curr_error_here--;
                }
            }

            // check if there are any newpoints that can be directed to the left
            while(curr_pointer<n&&X[indices_sorted[f0][curr_pointer]*m+f0]-dL[f0]<=threshold){
                int the_indice = indices_sorted[f0][curr_pointer];
                curr_pointer++;
                if(!datapoints[the_indice]){
                    continue;
                }
                bool current_classification = y[the_indice];
                // check if it is only on the left
                if(X[the_indice*m+f0]+dR[f0]<=threshold){
                    if(current_classification!=rightClassification&&current_classification==leftClassification){
                        curr_error_here--;
                    }
                    if (current_classification==rightClassification&&current_classification!=leftClassification){
                        curr_error_here++;
                    }
                    continue;
                }
                // on both sides
                if(current_classification!=leftClassification&&current_classification==rightClassification){
                    curr_error_here++;
                }
                q.enqueue(the_indice);
                
            }
            bestValue=min(bestValue,curr_error_here);
        }
    }

    return bestValue;

}
/*
    * Backtracking function for the robust decision tree problem.
    * This function explores the decision tree structure recursively, trying to find the optimal configuration.
    * It uses a depth-first search approach and prunes branches that cannot improve the current upper bound.
    * @param X The feature matrix.
    * @param y The labels of the data points.
    * @param dL Left perturbations per feature.
    * @param dR Right perturbations per feature.
    * @param s The current split index.
    * @param data_points_sorted The sorted data points.
    * @param upperbound The current upper bound for the error.
    * @param variables The current variables in the decision tree.
    * @param index_in_travel The index in the travel order of the decision nodes.
    * @param tree The current candidate trees.
    * @param indices_sorted The sorted indices of the data points per feature.
    * @return The minimum number of misclassified points found in this branch.
*/
int backtracking(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,vector<Variable2>& variables,int index_in_travel,T2& tree,vector<vector<int>>& indices_sorted){
        nr_of_calls++;
        if (checkTime()) {
            // Timeout!
            return infty;
        }
        if(curr_error>=upperbound->error){
            // Cannot improve the upper bound
            return curr_error;
        }

        int resValue = infty;

        // Get the index of the current variable in the travel order
        int index = travel_order[index_in_travel];

        if (index > 2){
            // All decision nodes have been traversed, we are at the leaves
        
            if (curr_error<upperbound->error){
                upperbound->error = curr_error;
                for(int i = 0 ; i<variables.size();++i){
                    upperbound->tree[i].chosenDim=variables[i].chosenDim;
                    upperbound->tree[i].chosenSplit=variables[i].chosenSplit;
                    upperbound->tree[i].isLeaf=variables[i].isLeaf;
                    upperbound->tree[i].classification=variables[i].classification;
                }
            }
            return curr_error;
        }
        Variable2& currentVariable =(variables[index]);
        if(index == 0){
            // We are at the root node, we need to find the best split for the root
            variables[0].chosenDim = tree.f0;
            int a = tree.a;
            int b = tree.b;
            int dim = tree.f0;
            int startPoint = 0;
            double startValue = data_points_sorted[dim][tree.a];
            // Decide the indices that are going to be considered for the root split.
            while(startPoint < n&&abs(X[m*indices_sorted[dim][startPoint]+dim]- startValue) > epsilon){
                startPoint++;
            }
            int endPoint = n-1;
            double endValue = data_points_sorted[dim][tree.b];
            while(endPoint>=0 && abs(X[m*indices_sorted[dim][endPoint]+dim]- endValue) > epsilon ){
                endPoint--;
            }
            
            vector<char>& curr_points_left = variables[1].datapoints;
            vector<char>& curr_points_right = variables[2].datapoints;
            // Queue for points that can be redirected to both sides
            Queue& clusterC = clusterCs[index];
            // Reset queue
            clusterC.count = 0;
            clusterC.front = 0;
            clusterC.rear = -1;
            for(int j = 0; j < n;++j){
                curr_points_right[j] = true;
                curr_points_left[j] = false;
            }
            int curr_pointer = 0;
            int skipsRemaining = 0;
            double prev_threshold = infty;
            // Start iterating over the sorted indices for the root split
            for(int i = startPoint; i <= endPoint; ++i){
                
                double threshold = X[indices_sorted[dim][i]*m+dim]+dR[dim]+epsilons_for_errors[dim];
                skipsRemaining--;
                // Either we already considered this threshold or neighbourhood pruning allows us to skip it
                if(skipsRemaining>=0|| abs(prev_threshold-threshold)<epsilons_for_errors[dim]){
                    continue;
                }
                variables[0].chosenSplit = threshold;
                prev_threshold=threshold;
                // Check if there are any points that were on boths sides that are now only on the left
                while(clusterC.size()>0&&Xtransposed[clusterC.peek()+dim*n]+dR[dim]<=threshold){
                    int indice = clusterC.peek();
                    clusterC.dequeue();
                    curr_points_left[indice] = true;
                    curr_points_right[indice] = false;
                }
                // Check if there are any new points that can be directed to the left
                while(curr_pointer<n&&Xtransposed[indices_sorted[dim][curr_pointer]+n*dim]-dL[dim]<=threshold){
                    int the_indice = indices_sorted[dim][curr_pointer];
                    curr_pointer++;
                    // Can only be directed to the left
                    if(Xtransposed[the_indice+n*dim]+dR[dim]<=threshold){
                        curr_points_left[the_indice] = true;
                        curr_points_right[the_indice] = false;
                        continue;
                    }
                    // Both sides
                    curr_points_left[the_indice] = true;
                    curr_points_right[the_indice] = true;
                    clusterC.enqueue(the_indice);
                }
                // Reset the current error
                curr_error = 0;
                
                int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
                
                skipsRemaining = backTrack-upperbound->error;

                resValue = min(backTrack,resValue);

            }
            return resValue;
        }
        
        vector<int> dims = tree.F1;
        // Right decision node
        if (index == 2){
            dims = tree.F2;
            // Reset the misclassifications in the right Node once we enter it anew
            for(int i = 0 ; i < n;++i){
                backup_index_2[i]= misclassified_per_datapoint[i];
            }
        }else{
            for(int i = 0 ; i < n;++i){
            // Reset the misclassifications in the left Node once we enter it anew
                backup_index_1[i]= misclassified_per_datapoint[i];
            }
        }
        bool isLeft = (index == 1);
        
        
        int back_up_error = curr_error;
        // Get the queue for the node
        Queue& clusterC = clusterCs[index];
        
        for(int dim : dims){
            // Refresh the queue
            clusterC.count = 0;
            clusterC.front = 0;
            clusterC.rear = -1;
            currentVariable.chosenDim = dim;
            int curr_pointer = 0;
            for(int i = 0 ; i < n ;++i){
                if(!currentVariable.datapoints[i]){
                    continue;
                }
                // initially all datapoints on the decision node are directed to the right leaf
                if(y[i]!=variables[2*index+2].classification){
                    bool was_empty = misclassified_per_datapoint[i]==0;
                    misclassified_per_datapoint[i]|=(1<<(2*index+2));
                    if(was_empty){
                        curr_error++;
                    }
                    
                }
            }
            currentVariable.chosenSplit = 0;
            // If the two leaves have the same classification, we can skip the rest of the search
            if(variables[2*index+1].classification == variables[2*index+2].classification){
                int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
                resValue = min(backTrack,resValue);
                // Reset the current error and the misclassifications
                curr_error = back_up_error;
                for(int i = 0 ; i < n;++i){
                    if(index == 1){
                        misclassified_per_datapoint[i] = backup_index_1[i];
                    }else{
                    
                        misclassified_per_datapoint[i] = backup_index_2[i];
                
                    }
                } 
                break;
            }
            int skipsRemaining = 0;
            double prev_thresh=infty;
            // Explore the thresholds for the current dimension
            for(int j = 0 ; j < n;++j){
                int i  = indices_sorted[dim][j];
                // The threshold value is set.
                double threshold = X[i*m+dim]+ epsilons_for_errors[dim]+dR[dim];
                if(!currentVariable.datapoints[i]){
                    continue;
                }
                skipsRemaining--;
                if(!((abs(threshold - prev_thresh )> epsilon&&skipsRemaining < 0))){
                    continue;
                }
                prev_thresh = threshold;
                currentVariable.chosenSplit = threshold;
                // Check if there are any points that were on boths sides that are now only on the left
                while(clusterC.size()>0&&Xtransposed[clusterC.peek()+dim*n]+dR[dim]<=threshold){
                    int the_indice = clusterC.peek();
                    clusterC.dequeue();
                    bool was_empty = misclassified_per_datapoint[the_indice]==0;
                    misclassified_per_datapoint[the_indice]&=(~(1<<(2*index+2)));
                    bool is_empty_now = misclassified_per_datapoint[the_indice]==0;
                    if((!was_empty) && is_empty_now){
                        curr_error--; 
                    }
                }
                // Check if there are any new points that can be directed to the left
                while(curr_pointer<n&&Xtransposed[indices_sorted[dim][curr_pointer]+n*dim]-dL[dim]<=threshold){
                    int the_indice = indices_sorted[dim][curr_pointer];
                    curr_pointer++;
                    if(!currentVariable.datapoints[the_indice]){
                        continue;
                    }
                    // Directly from right to left.
                    if(Xtransposed[the_indice+n*dim]+dR[dim]<=threshold){
                        int was_empty = misclassified_per_datapoint[the_indice]==0;
                        // Remove the misclassification from the right leaf
                        misclassified_per_datapoint[the_indice]&=(~(1<<(2*index+2)));
                        // Check if the point is misclassified in the left leaf
                        if(y[the_indice]!=variables[2*index+1].classification){
                            misclassified_per_datapoint[the_indice]|=(1<<(2*index+1));
                        }
                        int is_empty_now = misclassified_per_datapoint[the_indice]==0;
                        if((!was_empty) && (is_empty_now)){
                            curr_error--; 
                        }
                        // If it is newly misclassified in the left leaf

                        if((was_empty) && (!is_empty_now)){
                            curr_error++;
                        }
                        continue;
                    }
                    // in both sides
                    clusterC.enqueue(the_indice);
                    int was_empty = 0==misclassified_per_datapoint[the_indice];
                    if(y[the_indice]!=variables[2*index+1].classification){
                        // If the point is misclassified in the left leaf
                            misclassified_per_datapoint[the_indice]|=(1<<(2*index+1));
                    }
                    int is_empty_now = 0==misclassified_per_datapoint[the_indice];
                    // If it is newly misclassified in the left leaf
                    if(was_empty && !is_empty_now){
                        curr_error++;
                    }
                    
                }
                int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted);
                // For neighbourhood pruning as in Theorem 2.
                skipsRemaining = backTrack-upperbound->error;
                resValue = min(backTrack,resValue);
            }   
            // Refresh the current error and restore the misclassifications before the next iteration.
            curr_error = back_up_error;
            for(int i = 0 ; i < n;++i){
                if(index == 1){
                    misclassified_per_datapoint[i] = backup_index_1[i];
                }else{
                    misclassified_per_datapoint[i] = backup_index_2[i];
                }
            }   
            
        }
        if(resValue == infty){
            return upperbound->error;
        }
        return resValue;

}


int exhaustive_search_depth_2(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,T2& tree,vector<vector<int>>& indices_sorted,vector<Variable2>& variables){
    int min_value=infty;

    // Iterate over all possible classifications for the leaves
    for(int i = 0; i < 16; ++i){
        variables[3].classification = (i == 0) | (i == 1) | (i == 2) | (i == 3) | (i == 4) | (i == 5) | (i == 6) | (i == 7);
        variables[4].classification = (i == 0) | (i == 1) | (i == 2) | (i == 3) | (i == 8) | (i == 9) | (i == 10) | (i == 11);
        variables[5].classification = (i == 0) | (i == 1) | (i == 4) | (i == 5) | (i == 8) | (i == 9) | (i == 12) | (i == 13);
        variables[6].classification = (i == 0) | (i == 2) | (i == 4) | (i == 6) | (i == 8) | (i == 10) | (i == 12) | (i == 14);
        curr_error = 0;
        for(int j = 0 ; j < n ; ++j){
            misclassified_per_datapoint[j]=0;
        }
        min_value = min(min_value,backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,0,tree,indices_sorted));
        
    }
    return min_value;

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
        // All possible trees can be made from the initial candidates.
        AL2.push_back(T2(i,0,size-1,F1,F2));
    }
    // Sort features by the number of unique values they have.
    sort(AL2.begin(),AL2.end(),[&](T2 i,T2 j){return data_points_sorted[i.f0].size()<data_points_sorted[j.f0].size();});
    
    return AL2;
}
void QuantBNB2(vector<double>& X,vector<char>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,vector<vector<int>>& indices_sorted){
    // Initial candidate trees.
    vector<T2> ALk = prepareAL0(m,data_points_sorted);

    
    int current_size = ALk.size();
    vector<Variable2> sol(7);
    vector<Variable2> fakeSol(7);
    vector<Variable2> currSol(7);
    misclassified_per_datapoint = vector<int>(n,0);
    backup_index_1 = vector<int>(n,0);
    backup_index_2 = vector<int>(n,0);

    for(int i = 0; i<7;++i){
        bool isLeaf = i>2;
        sol[i] = Variable2(isLeaf);
        currSol[i] = Variable2(isLeaf);
    }

    currSol[1].datapoints = vector<char> (n,false);
    currSol[2].datapoints = vector<char> (n,false);
    UB* upper_bound = new UB(sol,warm_start_value);
    points_to_be_selected_right = vector<char>(n,false);
    clusterCs[0] = Queue(n);
    clusterCs[1] = Queue(n);
    clusterCs[2] = Queue(n);
    q = Queue(n);
    vector<int> oneElementF1(1,0);
    vector<int> oneElementF2(1,0);
    vector<int> F2Best(m,infty);
    T2 curr = T2(-1,-1,-1,oneElementF1,oneElementF2);
    int alkSize = ALk.size();
    vector<int> backTrackValuesAtsEquiSpaced;
    if(s<n){
        backTrackValuesAtsEquiSpaced= vector<int>(s+1,0);
    }
    while (true){
        
        if(record_progess && oldUpperBound!=upper_bound->error){
                // Record the progress if the upper bound has changed
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
        // Go through all the candidate trees in ALk
        for(int o = 0; o<alkSize;++o){
            alkSize = 0;
            T2 al = ALk.back();
            ALk.pop_back();            
            int f0 = al.f0;
            

            int a = al.a;
            int b = al.b;
            vector<int>& F1 = al.F1;
            vector<int>& F2 = al.F2;
            if (b-a<=s){
                // brute forcing
                exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,al,indices_sorted,currSol);
            }else{
                    // compute UB
                    
                            // let t_0,...,t_s be almost-s equi spaced
                    
                    for(int j = 0; j<=s; j++){
                        int tj = (a+(int)((j*1.0*(b-a)))/(1.0*s));
                        al.a=tj;
                        al.b=tj;
                        // Compute new upper bounds
                        int backTrack = exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,al,indices_sorted,currSol);
                        // Store the backtracking values for equi spaced points for neighbourhood pruning
                        backTrackValuesAtsEquiSpaced[j]=backTrack;
                    }
                    al.a = a;
                    al.b = b;
                    // let t_0,...,t_s be almost-s equi spaced in [a,b]
                    double lower_uf = 3;
                    for(int j = 1; j<=s; j++){
                        // Arrays for the surviving features in F1 and F2
                        vector<int> F1j= vector<int>();
                        vector<int> F2j= vector<int>();
                        int tj = (a+(int)((j*1.0*(b-a)))/(1.0*s));
                        if(tj>b){
                            break;
                        }
                        int tjmm = (a+(int)(((j-1)*1.0*(b-a)))/(1.0*s));

                        // Compute the interval for the lower bound as [t_{j-1},t_j]. We add epsilons to avoid numerical issues.
                        double lower_a = data_points_sorted[f0][tjmm]-epsilons_for_errors[f0];
                        double lower_b = data_points_sorted[f0][tj]+ epsilons_for_errors[f0]; 

                        for(int i = 0 ; i<m;++i){
                           F2Best[i]= infty;
                        }
                    
                        for(int f1: F1){
                            int maxError =infty;
                            for(int f2: F2){
                                if (checkTime()) {
                                    goto end;
                                }
                                // Compute lower bound by creating two depth 1 trees.
                                int l = solve_depth1_robust_efficient(y,data_points_sorted,points_to_be_selected_right,f1,indices_sorted,dR,dL,-1,lower_a,f0,X);
                                int r = solve_depth1_robust_efficient(y,data_points_sorted,points_to_be_selected_right,f2,indices_sorted,dR,dL,lower_b,lower_uf,f0,X);
                                int m =infty;
                                int new_a = tjmm;
                                int new_b = tj;
                                // Additional Step for a tighter bound. Can be removed
                                int s_prime = floor((0.6*n*s)/(1.0*(new_b)-(new_a)));
                                
                                // let t_0,...,t_s' be almost-s' equi spaced in [t_{j-1}+1,tj-1]
                                if(new_b-new_a>s_prime){
                                    for(int j_prime = 1; j_prime<=s_prime; j_prime++){
                                        // Calculate the interval of the middle part of the array that was dropped.
                                        int tj_prime = (new_a+(int)((j_prime*1.0*(new_b-new_a)))/(1.0*s_prime));
                                        int tjmm_prime = (new_a+(int)(((j_prime-1)*1.0*(new_b-new_a)))/(1.0*s_prime));
                                        double b_prime = data_points_sorted[f0][tj_prime]+epsilons_for_errors[f0];
                                        double a_prime = data_points_sorted[f0][tjmm_prime]-epsilons_for_errors[f0];
                                        int m_left = solve_depth1_robust_efficient(y,data_points_sorted,points_to_be_selected_right,f1,indices_sorted,dR,dL,lower_a,a_prime,f0,X);
                                        int m_right = solve_depth1_robust_efficient(y,data_points_sorted,points_to_be_selected_right,f2,indices_sorted,dR,dL,b_prime,lower_b,f0,X);
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
                            // Decide if the feature f1 is good enough to be added to the next ALk
                            if(maxError<upper_bound->error){
                                F1j.push_back(f1);
                            }else{
                                prunedF1++;
                            }
                        }
                        // Same for F2, we do it like this to not recompute the lower bounds for the same feature pairs.
                        for(int i = 0; i<m;++i){
                            if(F2Best[i]<upper_bound->error){
                                F2j.push_back(i);
                            }else{
                                prunedF2++;
                            }
                        }
                         
                        if(F1j.size()==0||F2j.size()==0){
                            continue;
                        }else{
                            // Apply neighbourhood pruning
                            double value_to_search = data_points_sorted[f0][tjmm];
                            int number_of_skips = max(0,backTrackValuesAtsEquiSpaced[j-1]-upper_bound->error);
                            int currPointer = 0;
                            while(currPointer<n&&abs(X[indices_sorted[f0][currPointer]*m+f0]-value_to_search)>epsilons_for_errors[f0]){
                                currPointer++;
                            }
                            currPointer= min(currPointer+number_of_skips,n-1);
                            double new_threshold = X[indices_sorted[f0][currPointer]*m+f0];
                            int currPointer_in_unique_values = 0;
                            while(currPointer_in_unique_values<n&&abs(data_points_sorted[f0][currPointer_in_unique_values]-new_threshold)>epsilons_for_errors[f0]){
                                currPointer_in_unique_values++;
                            }
                            // We can skip it if the full interval gets pruned
                            if(currPointer_in_unique_values>tj-1){
                                continue;
                            }
                            // Added to next $AL^k$
                            ALkpp.push_back(T2(f0,currPointer_in_unique_values,tj-1,F1j,F2j));                   
                        }
                    }                
            }
        }
        // Save the current ALkpp to the next ALk
        for(T2& e : ALkpp){
            ALk.push_back(e);
        }
        alkSize = ALk.size();
    }
 end:
    // Output the results
    cout<<"2";
    for(auto& e : upper_bound->tree){
        cout<<"\n"<<e;
    }
    cout<<"\n";
    // If record progress is enabled, print the timings and upper bounds
    for(int i = 0 ; i < upper_bounds.size();++i){
        cout << timings[i].count()<<" "<<upper_bounds[i]<<"\n";
    }
    // Terminated by a ***
    cout<<"***\n";
    // The rest is just logging information, not used in the algorithm
    cout<<"\nerror "<<upper_bound->error;
    cout<<"\nnumber of calls: "<<nr_of_calls<<"\n";
    cout<<"F1 pruned:"<<prunedF1<<"\n";
    cout<<"F2 pruned:"<<prunedF2<<"\n";
    // Except this, which is used to check if the time limit was exceeded
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
   

    /*
    * This section reads in the input from a file.
    */
    int index = 0;
    n = stoi(argv[index++]);
    m = stoi(argv[index++]);
    int misclassifiedCount = 0;
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
        for (size_t j = 0; j < sorted.size(); ++j) {
            if (result.size()==0||abs(sorted[j] - result.back()) > 1e-7) {
                result.push_back(sorted[j]);
            }
        }
        data_points_sorted[i] =result;
    }
    // Prepare indices sorted for each feature.
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
    // Initialize the epsilons for double errors. Extensible in the future for custom epsilon values per feature.
    for(int i = 0; i <m;++i){
        epsilons_for_errors[i] = epsilon;
    }
    QuantBNB2(X,Y,deltaL,deltaR,s,data_points_sorted,indices_sorted);
   
    
    

    return 0;
}
