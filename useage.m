close all;
clear;
% [TrainingTime_ELM, TestingTime_ELM, TrainingAccuracy_ELM, TestingAccuracy_ELM]=ELM('sinc_train', 'sinc_test', 0, 20, 'sig');
% [TrainingTime_SELM, TestingTime_SELM, TrainingAccuracy_SELM, TestingAccuracy_SELM]=SELM('sinc_train', 'sinc_test', 0, 20, 'sig');
iteration_times=10;
TrainingTime_SELM=Inf*ones(iteration_times,1);
TestingTime_SELM=Inf*ones(iteration_times,1);
TrainingAccuracy_SELM=Inf*ones(iteration_times,1);
TestingAccuracy_SELM=Inf*ones(iteration_times,1);
TrainingTime_ELM=Inf*ones(iteration_times,1);
TestingTime_ELM=Inf*ones(iteration_times,1);
TrainingAccuracy_ELM=Inf*ones(iteration_times,1);
TestingAccuracy_ELM=Inf*ones(iteration_times,1);
for i=1:iteration_times
    number_of_columns=i*1000;
    number_of_rows=500;
[TrainingTime_SELM(i), TestingTime_SELM(i), TrainingAccuracy_SELM(i), TestingAccuracy_SELM(i)]=SELM('diabetes_train', 'diabetes_test', 1, 40, 'sig',number_of_columns,number_of_rows);
disp(['The training time of SELM is ',num2str(TrainingTime_SELM(i))]);
[TrainingTime_ELM(i), TestingTime_ELM(i), TrainingAccuracy_ELM(i), TestingAccuracy_ELM(i)]=ELM('diabetes_train', 'diabetes_test', 1, 40, 'sig',number_of_columns,number_of_rows);
disp(['The training time of ELM is ',num2str(TrainingTime_ELM(i))]);
end
