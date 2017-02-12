function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,output_training,output_testing] = ELM_sparse_Input_Weight(train_data, test_data, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% [TrainingTime_ELM_proximal, TestingTime_ELM_proximal, TrainingAccuracy_ELM_proximal, TestingAccuracy_ELM_proximal,train_output_ELM_proximal,test_output_ELM_proximal]
iteration_times=10;
%This program is used to update the dictionary set and sparse
%representation through Proximal gradient descent method.
%The parameters we use are listed as follows:
%X: The input data
%K: The number of dictionary atoms
%lambda1: The trade-off of L2 or F norm
%lambda2: The trade-off of L1 norm
%opts: Another parameters.
%opts.tol: Thershold of error
%opts.maxit: The max iteration times
%opts.D0: The original dictionary set opts.Y0 = coeff;
%opts.yType = 0;
%learn a dictionary D from
%---- min_{D,Y} 0.5*||X-D*Y||_F^2 + mu*||Y||_1 --------------%
%---- subject to norm(D(:,j)) <= 1 for all j ----------------%
T=train_data(:,1:No_of_Output)';
P=train_data(:,No_of_Output+1:size(train_data,2))';%training_data
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
% test_data=load(TestingData_File);
TV.T=test_data(:,1:No_of_Output)';
TV.P=test_data(:,No_of_Output+1:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

% NumberofTrainingData=size(P,2);
% NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases
% start_time_train=clock;
tic;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
% BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
% clear P;                                            %   Release input of training data 
% ind=ones(1,NumberofTrainingData);
% BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
% tempH=tempH;%+BiasMatrix;

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
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=pinv(H') * T';
H=H';
T=T';
L=size(H,2);%The number of Hidden Neurons
M=size(T,2);%The dimension of beta1...betaL
beta=zeros(L,M);%Defination of beta
% X=P;
%==========================================================================
%Update the Input weights and output weight.
X=P;
temp_c=sum(abs(X),2);
for i=1:iteration_times
temp_b=sum(abs(beta));
temp_matrix=temp_c*temp_b*abs(transpose(beta))+abs(X)*abs(T)*abs(transpose(beta));
stepsize_w=sqrt(3)/18*norm(temp_matrix,'fro');
S=(beta*transpose(H*beta-T)).*transpose(H).*(1-transpose(H));
InputWeight=InputWeight-1/stepsize_w*S*P';
tempH=InputWeight*P;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
H=H';
step_size_beta=norm(H'*H,'fro');
beta=beta-1/step_size_beta*H'*(H*beta-T);
end
%==========================================================================
% L=size(H,2);%The number of Hidden Neurons
% h=H(:,1);
% M=size(T,2);%The dimension of beta1...betaL
% beta=zeros(L,M);%Defination of beta
% E=T-H*beta+h*beta(1,:);%error matrix
% B=2*transpose(E)*H(:,1);
% step=sum(H.^2);%stepsize
% buffer=h*beta(1,:);
% beta(1,:)=((1-2*transpose(h)*h/step(1)).*beta(1,:)+B(1,:)/step(1)).*transpose((B(1,:)~=0));
% position=repmat(1:L,1,iteration_times);
% for i=2:iteration_times
%     position_temp=position(i);
%     h=H(:,position_temp);
%     E=E-buffer+h*beta(position_temp,:);
%     B=2*transpose(E)*H(:,position_temp);
%     B=transpose(B);
%     beta(1,:)=((1-2*sum(h.^2)/step(position_temp)).*beta(position_temp,:)...
%         +B/step(position_temp)).*(B~=0);
%     buffer=h*beta(position_temp,:);
% end
TrainingTime=toc;
Y=(H * beta);                             %   Y: the actual output of the training data
TrainingAccuracy=sqrt(mse(T - Y)) ;              %   Calculate training accuracy (RMSE) for regression case
clear H;
output_training=Y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
% ind=ones(1,NumberofTestingData);
% BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
% tempH_test=tempH_test;% + BiasMatrix;
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
        %%%%%%%% More activation functions can be added here        
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TY=(H_test' * beta)';                       %   TY: the actual output of the testing data
% end_time_test=clock;
% TestingTime=etime(end_time_test,start_time_test);           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingTime=toc;
TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
output_testing=TY;
sparsity=sum(sum((InputWeight==0)));
disp(['The sparsity of input weight of Sparse input ELM is ',num2str(sparsity/numel(beta))]);