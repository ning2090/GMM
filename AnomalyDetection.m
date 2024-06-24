% Initialization初始化
clc; %清除命令窗口的内容
clear; %清除工作空间的所有变量
clf; %清除当前的Figure

% Generate two Gaussians
% data 1 参数定义
mu1 = [0 0];    sigma1 = [1 -0.9; -0.9 2]; %mu均值向量 sigma协方差
r1 = mvnrnd(mu1,sigma1, 100); % 生成100个二元正态分布随机数
% data 2
mu2 = [5 5];    sigma2 = [3 -2; -2 2];
r2 = mvnrnd(mu2,sigma2, 100);
% data 3
%mu3 = [2 3];    sigma3 = [3 2; 2 2];
%r3 = mvnrnd(mu3,sigma3, 100);
% Plot these Gaussians 
figure(1)
%subplot(2,1,i);
%title('anomaly data');
plot(r1(:,1),r1(:,2),'g+'); 
hold on; 
plot(r2(:,1),r2(:,2),'g+');
%plot(r3(:,1),r3(:,2),'g+');
plot(-4,-4,'r+');
plot(-3,9,'r+');
plot(11,11,'r+');
plot(10.8,10.5,'r+');
plot(-5,10.1,'r+');
plot(-3.1,9.6,'r+');
title('anomaly data');
axis([-10 15 -10 15]) %设置坐标轴范围和纵横比
hold off; 
