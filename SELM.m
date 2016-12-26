function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = SELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,scale_dim,scale_length)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
snr=50;
power_signal=1;
seed=0;
% scale_dim=1;
% scale_length=1;
threshold=0.15;
iteration_times=1;
% threshold_of_var=0.00005;
REGRESSION=0;
CLASSIFIER=1;
%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
train_data=repmat(train_data,scale_dim,scale_length);
disp(['The size of Training data is ',num2str(size(train_data,1)),' X ',num2str(size(train_data,2))]);
% train_data=awgn(train_data,snr,power_signal,seed);
T=train_data(:,1)'; %First column are the expected output (target) for regression and classification applications
P=train_data(:,2:size(train_data,2))';%Input data, the rest columns consist of different attributes information of each instance
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
test_data=repmat(test_data,scale_dim,scale_length);
% test_data=awgn(test_data,20);
TV.T=test_data(:,1)';%First column are the expected output (target) for regression and classification applications
TV.P=test_data(:,2:size(test_data,2))';%the rest columns consist of different attributes information of each instance
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);%This is the number of training data
NumberofTestingData=size(TV.P,2);%This is the number of testing data
NumberofInputNeurons=size(P,1);%This is dimention of input data

if Elm_Type~=REGRESSION %ELM_type is 0 for regression; 1 for (both binary and multi-classes) classification
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);% cat(2,T,TV.T) generate the matrix [T,TV.T]. sort(*,2)sorts the elements in each row.
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;%Initialization of j
    for i = 2:(NumberofTrainingData+NumberofTestingData)%Calcualte the number of cluster to classify
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);%Select the first element of sorted_target and compare with the element in sorted_target, if there are some elements different from the first element of sort_target. The label is changed to the next element in sort_target
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
       
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)%First column are the expected output (target) for regression and classification applications
                break; %If the ith element in T is equal to the jth element of label, the i column and j row is 1 and another elements in jth column is 0
            end
        end
        temp_T(j,i)=1;%If the ith element in T is equal to the jth element of label, the i column and j row is 1 and another elements in jth column is 0
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)%First column are the expected output (target) of testing data
                break; %If the ith element in T is equal to the jth element of label, the i column and j row is 1 and another elements in jth column is 0
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 %   end if of Elm_Type
OutputWeight=ones(NumberofHiddenNeurons,size(T,1));
I=1:NumberofHiddenNeurons;
flag=0;
iteration_break=NumberofHiddenNeurons;
%%%%%%%%%%% Calculate weights & biases
% start_time_train=cputime;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
start_time_training=clock;
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;%P is the training data
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);%equals to repmat(BiasofHiddenNeurons,1,NumberofTrainingData)              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;%InputWeight*P+BiasMatrix->input_weight*training_data+bias
%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=repmat(median(median(T))/median(median(H)),size(H,1),size(T,1));
% OutputWeight=repmat(transpose(mean(T,2)),size(H,1),1)./repmat(mean(H,2),1,size(T,1));
for iteration=1:iteration_times
    former_OutputWeight=OutputWeight;
    for i=1:NumberofHiddenNeurons
        if i==iteration_break+1
            disp('break when i=');
            disp(i);
            break;
        end
        error_matrix=T-(H' * OutputWeight)'+(OutputWeight(I(i),:))'*(H(I(i),:));
        OutputWeight(I(i),:)=H(I(i),:)*error_matrix'/(H(I(i),:)*(H(I(i),:))');
        Y=(H' * OutputWeight)';  
        if sqrt(mse(T - Y))/size(T,1)<=threshold;
            flag=1;
            disp('FLAG is 1 when');
            disp('iteration=');
            disp(iteration);
            disp('1=');
            disp(i);
            break;
        end
    end
    if flag==1
        break;
    end
    [~,I]=sort(var(former_OutputWeight-OutputWeight,0,2),1);
%     iteration_break=sum(B>=NumberofHiddenNeurons*threshold_of_var);
end
% disp('iteration=');disp(iteration);
% disp('i=');disp(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for iteration=1:iteration_times
%     for i=1:NumberofHiddenNeurons
%         error_matrix=T-(H' * OutputWeight)'+(OutputWeight(I(i),:))'*(H(I(i),:));
%         OutputWeight(I(i),:)=H(I(i),:)*error_matrix'/(H(I(i),:)*(H(I(i),:))');
%         Y=(H' * OutputWeight)';  
%         if sqrt(mse(T - Y))/size(T,1)<=threshold;
%             disp('FLAG is 1 when');
%             disp('iteration=');
%             disp(iteration);
%             disp('1=');
%             disp(i);
%             break;
%         end
%     end
% end

%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 

%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

% end_time_train=cputime;
% TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM
end_time_training=clock;
TrainingTime=etime(end_time_training,start_time_training);

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
% start_time_test=cputime;
start_time_testing=clock;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
% end_time_test=cputime;
end_time_testing=clock;
% end_time_test=toc(timerVal);
% TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingTime=etime(end_time_testing,start_time_testing);
if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  
end