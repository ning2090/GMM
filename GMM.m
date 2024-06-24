% Initialization初始化
clc; %清除命令窗口的内容
clear; %清除工作空间的所有变量
clf; %清除当前的Figure

% Generate two Gaussians
% data 1 参数定义
mu1 = [0 0];    sigma1 = [2 -0.9; -0.9 2]; %mu均值向量 sigma协方差
r1 = mvnrnd(mu1,sigma1, 100); % 生成100个二元正态分布随机数
% data 2
%mu2 = [5 5];    sigma2 = [3 -2; -2 2];
%r2 = mvnrnd(mu2,sigma2, 100);
% data 3
mu2 = [5 3];    sigma2 = [3 2; 2 2];
r2 = mvnrnd(mu2,sigma2, 100);
% Plot these Gaussians 
figure(1)
for i = 1:2
    subplot(2,2,i); %将图形窗口分成2行2列的子图网格，通过i来指定当前子图的位置
    title('Original data');
    plot(r1(:,1),r1(:,2),'r+'); %以红色加号为标记符号，将r1中的数据点绘制在二维坐标轴上
    hold on; %保留住前图的坐标和图形，将新绘制的图形共同呈现在该图上，并自动调整坐标轴的范围
    plot(r2(:,1),r2(:,2),'b+');
    %plot(r3(:,1),r3(:,2),'g+');
    title('Original data');
    %axis([-10 15 -10 15]) 设置坐标轴范围和纵横比
    hold off; %绘制新图，取消原图
end
data = [r1; r2]; % Our dataset
%data = [r1; r2; r3]; % Our dataset

% Do GMM fitting process拟合过程
fit_gmm(data,2,0.1);

%% Using Gaussian Mixture Model to do clustering
% Input:    data        - data,
%           k           - the number of Gaaussians,
%           threshold   - the precision of the stopping threshold
% Output:   lambda      - the weight for Gaussians
%           mu          - the means for Gaussians
%           sigma       - the covariance matrix for Gaussians
function [lambda, mu, sigma] = fit_gmm(data, k, precision)

    [num,dim] = size(data);   % Get the size and dimension of data尺寸和维度
    lambda = repmat(1/k,k,1); % Initialize weight for k-th Gaussian to 1/k
    
    randIdx = randperm(num);   % do randomly permutation process随机生成长度为num的随机排列数组
    mu = data(randIdx(1:k),:); % Initialize k means for Gaaussians randomly 
    
    dataVariance =  cov(data,1);    % Obtain the variance of dataset ∑(x-mu)'*(x-mu)/ num 求协方差
    sigma = cell (1, k);            % Store covariance matrices
    % sigma is initialized as the covariance of the whole dataset
    for i = 1 : k
        sigma{i} = dataVariance;
    end
    % x,y is used to draw pdf of Gaussians
    x=-5:0.05:10;%创建一个向量x，该向量从-5开始，以0.05为步长递增，直到10为止
    y=-5:0.05:10;
    
    iter = 0; precious_L = 100000;
    while iter < 100
        
        % E-step (Expectation)
        gauss = zeros(num, k); % 创建一个num×k的全零矩阵gauss，num为数据点的数量，k为高斯模型的数量。
        for idx = 1: k
            gauss(:,idx) = lambda(idx)*mvnpdf(data, mu(idx,:), sigma{idx});%计算每个数据点在该高斯模型下的概率密度函数值，并将结果保存在 gauss 矩阵中的对应位置
        end
        respons = zeros(num, k); %用于存储每个数据点对每个高斯模型的响应度
        
        total = sum(gauss, 2); %对gauss矩阵的每一行进行求和操作
        for idx = 1:num
            respons(idx, :) = gauss(idx,:) ./ total(idx); %将每个数据点对每个高斯模型的响应度进行归一化处理，./ 表示逐元素除法操作
        end

       % M-step (Maximization)
       responsSumedRow = sum(respons,1);
       responsSumedAll = sum(responsSumedRow,2);
       for i = 1 : k
          % Updata lambda
          lambda(i) =  responsSumedRow(i) / responsSumedAll;
          
          % Updata mu
          newMu = zeros(1, dim);
          for j = 1 : num
              newMu = newMu + respons(j,i) * data(j,:);
          end
          mu(i,:) = newMu ./ responsSumedRow(i);
          
          % Updata sigma
          newSigma = zeros(dim, dim);
          for j = 1 : num
              diff = data(j,:) - mu(i,:);
              diff = respons(j,i) * (diff'* diff);
              newSigma = newSigma + diff;
          end
          sigma{i} = newSigma ./ responsSumedRow(i);
       end
       
        subplot(2,2,2)
        title('Expectation Maxmization');
        hold on
        [X,Y]=meshgrid(x,y); %基于向量x和y中包含的坐标返回二维网格坐标

        stepHandles = cell(1,k);
        ZTot = zeros(size(X));

        for idx = 1 : k
           Z = getGauss(mu(idx,:), sigma{idx}, X, Y); %getGauss函数用于计算高斯分布的概率密度值。
           Z = lambda(idx)*Z;
           [~,stepHandles{idx}] = contour(X,Y,Z); %通过contour函数绘制轮廓曲线，contour函数用于绘制等高线图。[~,stepHandles{idx}]表示只获取轮廓线的句柄，并将该句柄保存在stepHandles数组的对应位置。
           ZTot = ZTot + Z; %将当前高斯模型的响应度Z加到ZTot矩阵
        end
        hold off
       
        subplot(2,2,3) % image 3 - PDF for GMM
        mesh(X,Y,ZTot),title('PDF for GMM');%创建一个网格图，该网格图为三维曲面

        subplot(2,2,4) % image 4 - Projection of GMM
        surf(X,Y,ZTot),view(2),title('Projection of GMM')%GMM投影。用于绘制三维曲面图。根据 `ZTot` 的值给曲面上的网格点染色，呈现出三维曲面的形状和颜色。`view(2)` 用于将图形的视角设置为从正上方观察
        shading interp %shading interp 是用于改变曲面绘制着色方式的函数，使用插值的方式对曲面进行着色，使得颜色变化更加平滑过渡
        %colorbar彩条
        drawnow();%用于强制刷新并重绘图形窗口的函数，每次迭代都实时显示最新的结果
       
       % Compute the log likelihood L
       temp = zeros(num, k);
       for idx = 1 : k
          temp(:,idx) = lambda(idx) *mvnpdf(data, mu(idx, :), sigma{idx}); 
       end
       temp = sum(temp,2);
       temp = log(temp);
       L = sum(temp);
       
       iter = iter + 1;
       preciousTemp = abs(L-precious_L)
       if  preciousTemp < precision
           break;
       else
           % delete plot handles in image 2
           for idx = 1 : k %for循环会遍历每个高斯模型，将所有轮廓曲线的线条颜色设置为不显示线条
                set(stepHandles{idx},'LineColor','none')
           end
       end
       precious_L = L;
       
    end
end 

function [Z] = getGauss(mean, sigma, X, Y)
    dim = length(mean);

    weight = 1/sqrt((2*pi).^dim * det(sigma));
    [~,row] = size(X);
    [~,col] = size(Y);
    Z = zeros(row, col);
    for i = 1 : row
        sampledData = [X(i,:); Y(i,:)]';
        sampleDiff = sampledData - mean;
        inner = -0.5 * (sampleDiff / sigma .* sampleDiff);
        Z(i,:) =  weight * exp(sum(inner,2));
    end

end