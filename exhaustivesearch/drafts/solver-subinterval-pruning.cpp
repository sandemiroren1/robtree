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
#include "sophisticated_struct.h"   
#include <iomanip>
#include <chrono>
#include <queue>
#include <map>

vector<bool> misclassified;
vector<int> revert_stack;
vector<bool> points_to_be_selected_left;
vector<bool> points_to_be_selected_right;
vector<int> travel_order= {0,1,3,4,2,5,6};

vector<int> stack_for_reversions;
int timelimit = 1<<30;
vector<map<int,pair<int,int>>> V_per_feature;
vector<set<int>> visited_per_feature;
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
bool checkTime(){
    if ((chrono::steady_clock().now() - start) > timelimit_s) {
        return true;
    }
    return false;
}
queue<int> q;

int current_working_theta_L;
int current_working_theta_R;

int current_working_theta_LL;
int current_working_theta_RR;


int misclassified_leaf_3;
int misclassified_leaf_4;
int misclassified_leaf_5;
int misclassified_leaf_6;

int solve_depth1_robust_efficient(vector<bool>& y,int s,vector<vector<double>>& data_points_sorted
    ,vector<bool>& datapoints,int f0,vector<vector<int>>&indices_sorted,vector<double>& dR
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

int handle_new_misclassifications(vector<vector<double>>& data_points_sorted,vector<Variable2>& variables, 
        bool chosen_class,vector<bool>& Y,vector<double>& dL,vector<double>& dR
        ,int index,int upperbound,vector<vector<int>>& indices_sorted){
    int newWrongs = 0;
    int parentIndex = (index-1)>>1;
    int parent_parentIndex = (parentIndex-1)>>1;
    if(index==3){
        current_working_theta_LL = 0;
    }
    if(index == 4){
        current_working_theta_L = current_working_theta_LL;
    }
    if(index == 5){
        current_working_theta_R = 0;
    }
    if(index == 6){
        current_working_theta_RR = current_working_theta_R;
    }
    const Variable2& parentVariable = variables[parentIndex];
    const Variable2& parentParentVariable = variables[parent_parentIndex];

    bool isLeftOfParent = (((parentIndex<<1)+1) == index);
    bool isLeftOfParentParent = (((parent_parentIndex<<1)+1) == parentIndex);

    const double dRParent = dR[parentVariable.chosenDim];
    const double dLParent = dL[parentVariable.chosenDim];
    const double dRParentParent = dR[parentParentVariable.chosenDim];
    const double dLParentParent = dL[parentParentVariable.chosenDim];

    const double parentSplit = parentVariable.chosenSplit;
    const double parentParentSplit = parentParentVariable.chosenSplit;

    const int parentDim = parentVariable.chosenDim;
    const int parentParentDim = parentParentVariable.chosenDim;
    const vector<int>& sorted_indices = indices_sorted[parentDim];
    if (isLeftOfParent){
        for(int j = 0; j<n;++j){

            int i = sorted_indices[j];
            
            bool continue_with_work = !(Y[i] == chosen_class );
            
            double theValue = Xtransposed[n*parentDim+i];
            double theValue2 = Xtransposed[n*parentParentDim+i];

            bool is_data_included_p1 = (theValue - dLParent <= parentSplit); 
            j = ((!is_data_included_p1)*(1<<20))+j;        
            
            if(continue_with_work&&is_data_included_p1&&(((isLeftOfParentParent && theValue2 - dLParentParent <= parentParentSplit) 
                            || (!isLeftOfParentParent && theValue2 + dRParentParent > parentParentSplit)))){
                if(theValue2>parentParentSplit&&index==5){
                    current_working_theta_R++;
                }
                if(theValue<=parentSplit&&index==3){
                    current_working_theta_LL++;
                }
            }
            if(continue_with_work&&is_data_included_p1&&!misclassified[i]&&(((isLeftOfParentParent && theValue2 - dLParentParent <= parentParentSplit) 
                            || (!isLeftOfParentParent && theValue2 + dRParentParent > parentParentSplit)))){
                misclassifiedCount++;
                newWrongs++;
                revert_stack[revert_stack_size++] = i;
                misclassified[i] = true;
            }
        }
    }else{
        for(int j = n-1; j>=0;j--){
        
            int i = indices_sorted[parentDim][j];

            bool continue_with_work = !(Y[i] == chosen_class);

            double theValue = Xtransposed[n*parentDim+i];
            double theValue2 = Xtransposed[n*parentParentDim+i];

            bool is_data_included_p1 = (theValue + dRParent > parentSplit);
            j = j-((!is_data_included_p1)*(1<<20));              
            if(continue_with_work&&is_data_included_p1  &&(((isLeftOfParentParent && theValue2 - dLParentParent <= parentParentSplit) 
                            || (!isLeftOfParentParent && theValue2 + dRParentParent > parentParentSplit)))){
                if(theValue>parentSplit&&index==6){
                    current_working_theta_RR++;
                    current_working_theta_R++;

                }
                if(theValue2<=parentParentSplit&&index==4){
                    current_working_theta_L++;
                }
            }
            if(continue_with_work&&is_data_included_p1&&!misclassified[i]&&(((isLeftOfParentParent && theValue2 - dLParentParent <= parentParentSplit) 
                            || (!isLeftOfParentParent && theValue2 + dRParentParent > parentParentSplit)))){
            
                misclassifiedCount++;
                newWrongs++;
                revert_stack[revert_stack_size++] = i;
                misclassified[i] = true;
            }
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

vector<double> chosen_f0equal0_thresholds;
int backtracking(vector<double>& X,vector<bool>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,vector<Variable2>& variables,int index_in_travel,T2& tree,vector<vector<int>>& indices_sorted,UB* best_for_this_time){
        nr_of_calls++;
        const int upperbound_error = upperbound->error;
        if (checkTime()) {
            return 1<<30;
        }
        //if(misclassifiedCount>=upperbound->error){
          //  return misclassifiedCount;
        //}
        int resValue = 1<<30;

        if (variables.size()==index_in_travel){
            best_for_this_time->current_working_theta_L=min(best_for_this_time->current_working_theta_L,current_working_theta_L);
                best_for_this_time->current_working_theta_LL=min(best_for_this_time->current_working_theta_LL,current_working_theta_LL);
                best_for_this_time->current_working_theta_R=min(best_for_this_time->current_working_theta_R,current_working_theta_R);
                best_for_this_time->current_working_theta_RR=min(best_for_this_time->current_working_theta_RR,current_working_theta_RR);
            if (misclassifiedCount< best_for_this_time->error){
                best_for_this_time->error = misclassifiedCount;
                for(int i = 0 ; i<variables.size();++i){
                    best_for_this_time->tree[i].chosenDim=variables[i].chosenDim;
                    best_for_this_time->tree[i].chosenSplit=variables[i].chosenSplit;
                    best_for_this_time->tree[i].isLeaf=variables[i].isLeaf;
                    best_for_this_time->tree[i].classification=variables[i].classification;
                }
                
            }
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
            int startPoint = 0;
            int dim = tree.f0;
            double startValue = data_points_sorted[dim][tree.a];
            while(startPoint < n&&abs(X[m*indices_sorted[dim][startPoint]+dim]- startValue) > epsilon){
                startPoint++;
            }
            int endPoint = n-1;
            double endValue = data_points_sorted[dim][tree.b];
            while(endPoint>=0 && abs(X[m*indices_sorted[dim][endPoint]+dim]- endValue) > epsilon ){
                endPoint--;
            }

            map<int,pair<int,int>>& V= V_per_feature[tree.f0]; // BST
            vector<pair<int,int>> Q;
            Q.push_back({startPoint,endPoint});
            set<int>& visited_map = visited_per_feature[tree.f0];

            while (Q.size()>0){
                pair<int,int> ij = Q.back();
                Q.pop_back();
                
                int i = ij.first;
                int j = ij.second;
                if (j<i){
                    continue;
                }
                int w = (i+j)>>1;
                int currIndice = indices_sorted[tree.f0][w];
                double data_point = X[m*currIndice+tree.f0];
                auto key_find = visited_map.find(int(1e6*data_point));
                if(key_find!=visited_map.end()){
                    Q.push_back({i,w-1});
                    Q.push_back({w+1,j});
                    continue;
                }
                
                auto u = V.lower_bound(w);
                auto v = V.upper_bound(w);

                double theta_u_L = 0;
                double theta_v_R = 0;
                //cout<<"*************\n";
                if(u!=V.end()){
                    theta_u_L = u->second.first;
                    //cout<<"theta_u: ("<<u->second.first<<","<<u->second.second<<")\n";

                }
                if(v!=V.end()){
                    theta_v_R = v->second.second;
                    //cout<<"theta_v: ("<<v->second.first<<","<<v->second.second<<")\n";
                }
                
                if(theta_u_L+theta_v_R>upperbound->error){
                    //cout<<"Pruned: "<<i<<" , "<<j<<" \n";
                    //cout<<"u_theta_L: "<<theta_u_L<<" , "<<v->second.second<<" \n";
                    
                    continue;
                }
                if(tree.f0==0){
                    chosen_f0equal0_thresholds.push_back(1e6*data_point);
                }
                currentVariable.chosenSplit = X[m*currIndice+tree.f0]+epsilons_for_errors[tree.f0]+dR[tree.f0];
                currentVariable.chosenDim = tree.f0;
                best_for_this_time->error=1<<30;
                best_for_this_time->current_working_theta_L=1<<30;
                best_for_this_time->current_working_theta_LL=1<<30;
                best_for_this_time->current_working_theta_R=1<<30;
                best_for_this_time->current_working_theta_RR=1<<30;
                int backTrack =backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted,best_for_this_time);
                resValue = min(resValue,backTrack);
                int increment = max(1,backTrack-upperbound->error);
                Q.push_back({i,w-1});
                Q.push_back({w+increment,j});
                V.insert({w,{best_for_this_time->current_working_theta_L,best_for_this_time->current_working_theta_R}});
                visited_map.insert(int(1e6*data_point));
            }
            
            
            return resValue;
        }
        if (index == 4||index==6){
            //cout<<"LEAF\n";
            currentVariable.classification = !variables[index-1].classification;
            int newWrongs = handle_new_misclassifications(data_points_sorted,variables,currentVariable.classification,y,dL,dR,index,upperbound_error,indices_sorted);
            resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted,best_for_this_time);
            revert_misclassifications(newWrongs);
            return resValue;
        }
        if (index == 3||index==5){
            //cout<<"LEAF\n";
            currentVariable.classification = false;
            int newWrongs = handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,index,upperbound_error,indices_sorted);
            resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted,best_for_this_time);
            revert_misclassifications(newWrongs);
            currentVariable.classification = true;
            newWrongs = handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,index,upperbound->error,indices_sorted);
            int backTrack = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted,best_for_this_time);
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

        variables[2*index+1].classification=false;
        variables[2*index+2].classification=false;
        currentVariable.chosenDim=0;
        currentVariable.chosenSplit=0;
        int newWrongs =  handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,2*index+1,upperbound_error,indices_sorted);
        newWrongs +=     handle_new_misclassifications(data_points_sorted,variables,false,y,dL,dR,2*index+2,upperbound_error,indices_sorted);
        resValue = backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+3,tree,indices_sorted,best_for_this_time);
        revert_misclassifications(newWrongs);
        variables[2*index+1].classification=true;
        variables[2*index+2].classification=true;
        newWrongs =   handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,2*index+1,upperbound_error,indices_sorted);
        newWrongs +=  handle_new_misclassifications(data_points_sorted,variables,true,y,dL,dR,2*index+2,upperbound_error,indices_sorted);
        resValue = min(resValue,backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+3,tree,indices_sorted,best_for_this_time));
        revert_misclassifications(newWrongs);

        for(int dim : dims){
            
                
            map<int,pair<int,int>> V; // BST
            vector<pair<int,int>> Q;
            set<int> visited_map;
            Q.push_back({0,n-1});
            currentVariable.chosenDim = dim;

            while (Q.size()>0){
                pair<int,int> ij = Q.back();
                Q.pop_back();
                
                int i = ij.first;
                int j = ij.second;
                if (j<i){
                    continue;
                }
                int w = (i+j)/2;
                int currIndice = indices_sorted[dim][w];
                double data_point = X[m*currIndice+dim];
                set<int> visited_map;
                auto key_find = visited_map.find(int(data_point*1e6));
                if(key_find!=visited_map.end()){
                    Q.push_back({i,w-1});
                    Q.push_back({w+1,j});
                    continue;
                }
                auto u = V.lower_bound(w);
                auto v = V.upper_bound(w);

                int theta_u_L = 0;
                int theta_v_R = 0;
                if(u!=V.end()){
                    theta_u_L = u->second.first;
                }
                if(v!=V.end()){
                    theta_v_R = v->second.second;
                }
                if(theta_u_L+theta_v_R>upperbound->error){
                    continue;
                }

                bool valid_split = isLeft&&X[currIndice*m+variables[0].chosenDim] - dL[variables[0].chosenDim] <= variables[0].chosenSplit;
                valid_split = valid_split || (!isLeft && X[currIndice*m+variables[0].chosenDim] + dR[variables[0].chosenDim] > variables[0].chosenSplit);
                int backTrack=1<<30;
                if(valid_split){
                    currentVariable.chosenSplit = X[currIndice*m+dim]+epsilons_for_errors[dim]+dR[dim];
                    backTrack =backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,index_in_travel+1,tree,indices_sorted,best_for_this_time);
                    resValue = min(resValue,backTrack);
                    int increment = max(1,backTrack-upperbound->error);
                    Q.push_back({i,w-1});
                    Q.push_back({w+increment,j});
                    V.insert({w,{best_for_this_time->current_working_theta_LL,best_for_this_time->current_working_theta_RR}});
                    visited_map.insert(int(1e6*data_point));
                }else{
                    Q.push_back({i,w-1});
                    Q.push_back({w+1,j});
                }
                
                



            }

        }
        if(index==1){

        }
        return resValue;

}


void exhaustive_search_depth_2(vector<double>& X,vector<bool>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,
    UB* upperbound,T2& tree,vector<vector<int>>& indices_sorted,vector<Variable2>& variables,UB* current_sol){
    misclassifiedCount = 0;
    revert_stack_size = 0;
    
    backtracking(X,y,dL,dR,s,data_points_sorted,upperbound,variables,0,tree,indices_sorted,current_sol);

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
        AL2.push_back(T2(i,0,size-1,F1,F2));
    }
    
    return AL2;
}
void QuantBNB2(vector<double>& X,vector<bool>& y,vector<double>& dL, vector<double>& dR,int s,vector<vector<double>>& data_points_sorted,vector<vector<int>>& indices_sorted){
    vector<T2> ALk = prepareAL0(m,data_points_sorted);
    int current_size = ALk.size();
    vector<Variable2> sol(7);
    vector<Variable2> fakeSol(7);
    vector<Variable2> currSol(7);
    stack_for_reversions= vector<int>(n+10,-1);
    for(int i = 0; i<7;++i){
        bool isLeaf = i>2;
        fakeSol[i] = Variable2(isLeaf);
        sol[i] = Variable2(isLeaf);
        currSol[i] = Variable2(isLeaf);
    }
    UB* current_bound = new UB(fakeSol,1<<30);
    UB* upper_bound = new UB(sol,warm_start_value);
    revert_stack = vector<int>(n,-1);
    misclassified = vector<bool>(n,false);
    points_to_be_selected_right = vector<bool>(n,false);

    vector<int> oneElementF1(1,0);
    vector<int> oneElementF2(1,0);
    vector<int> F2Best(m,1<<30);
    T2 curr = T2(-1,-1,-1,oneElementF1,oneElementF2);
    int alkSize = ALk.size();
    while (true){
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
                exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,al,indices_sorted,currSol,current_bound);
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
                                exhaustive_search_depth_2(X,y,dL,dR,s,data_points_sorted,upper_bound,curr,indices_sorted,currSol,current_bound);
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
                            ALkpp.push_back(T2(f0,tjmm,tj,F1j,F2j));                   
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

    //outfile<<"\nerror "<<upper_bound->error;
    cout<<"\nerror "<<upper_bound->error;
    cout<<"\nnumber of calls: "<<nr_of_calls<<"\n";
    if(!checkTime()){
        cout<<"True";
    }else{
        cout<<"False";
    }
    cout<<" Size chosen thresh:"<<chosen_f0equal0_thresholds.size()<<" size uniqued: "<<data_points_sorted[0].size()<<"\n";
    sort(chosen_f0equal0_thresholds.begin(),chosen_f0equal0_thresholds.end());
    for(auto e : chosen_f0equal0_thresholds){
        cout<<e<<"\n";
    }

    


}


int main(int argc, char* argv0[]) {
    // First argument is the filename
    string filename = argv0[1];
    cout<<setprecision(17);

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
    vector<bool> misclassified = vector<bool>(n,false);
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
    vector<double> X = vector<double>(n*m,0);
    Xtransposed = vector<double>(m*n);
    vector<bool> Y = vector<bool>(n);
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
    V_per_feature = vector<map<int,pair<int,int>>>(m,map<int,pair<int,int>>());
    visited_per_feature = vector<set<int>>(m,set<int>());
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
    for(auto e: visited_per_feature[0]){
        cout<<e<<"\n";
    }
    
    

    return 0;
}
