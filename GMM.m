% Initialization��ʼ��
clc; %�������ڵ�����
clear; %��������ռ�����б���
clf; %�����ǰ��Figure

% Generate two Gaussians
% data 1 ��������
mu1 = [0 0];    sigma1 = [2 -0.9; -0.9 2]; %mu��ֵ���� sigmaЭ����
r1 = mvnrnd(mu1,sigma1, 100); % ����100����Ԫ��̬�ֲ������
% data 2
%mu2 = [5 5];    sigma2 = [3 -2; -2 2];
%r2 = mvnrnd(mu2,sigma2, 100);
% data 3
mu2 = [5 3];    sigma2 = [3 2; 2 2];
r2 = mvnrnd(mu2,sigma2, 100);
% Plot these Gaussians 
figure(1)
for i = 1:2
    subplot(2,2,i); %��ͼ�δ��ڷֳ�2��2�е���ͼ����ͨ��i��ָ����ǰ��ͼ��λ��
    title('Original data');
    plot(r1(:,1),r1(:,2),'r+'); %�Ժ�ɫ�Ӻ�Ϊ��Ƿ��ţ���r1�е����ݵ�����ڶ�ά��������
    hold on; %����סǰͼ�������ͼ�Σ����»��Ƶ�ͼ�ι�ͬ�����ڸ�ͼ�ϣ����Զ�����������ķ�Χ
    plot(r2(:,1),r2(:,2),'b+');
    %plot(r3(:,1),r3(:,2),'g+');
    title('Original data');
    %axis([-10 15 -10 15]) ���������᷶Χ���ݺ��
    hold off; %������ͼ��ȡ��ԭͼ
end
data = [r1; r2]; % Our dataset
%data = [r1; r2; r3]; % Our dataset

% Do GMM fitting process��Ϲ���
fit_gmm(data,2,0.1);

%% Using Gaussian Mixture Model to do clustering
% Input:    data        - data,
%           k           - the number of Gaaussians,
%           threshold   - the precision of the stopping threshold
% Output:   lambda      - the weight for Gaussians
%           mu          - the means for Gaussians
%           sigma       - the covariance matrix for Gaussians
function [lambda, mu, sigma] = fit_gmm(data, k, precision)

    [num,dim] = size(data);   % Get the size and dimension of data�ߴ��ά��
    lambda = repmat(1/k,k,1); % Initialize weight for k-th Gaussian to 1/k
    
    randIdx = randperm(num);   % do randomly permutation process������ɳ���Ϊnum�������������
    mu = data(randIdx(1:k),:); % Initialize k means for Gaaussians randomly 
    
    dataVariance =  cov(data,1);    % Obtain the variance of dataset ��(x-mu)'*(x-mu)/ num ��Э����
    sigma = cell (1, k);            % Store covariance matrices
    % sigma is initialized as the covariance of the whole dataset
    for i = 1 : k
        sigma{i} = dataVariance;
    end
    % x,y is used to draw pdf of Gaussians
    x=-5:0.05:10;%����һ������x����������-5��ʼ����0.05Ϊ����������ֱ��10Ϊֹ
    y=-5:0.05:10;
    
    iter = 0; precious_L = 100000;
    while iter < 100
        
        % E-step (Expectation)
        gauss = zeros(num, k); % ����һ��num��k��ȫ�����gauss��numΪ���ݵ��������kΪ��˹ģ�͵�������
        for idx = 1: k
            gauss(:,idx) = lambda(idx)*mvnpdf(data, mu(idx,:), sigma{idx});%����ÿ�����ݵ��ڸø�˹ģ���µĸ����ܶȺ���ֵ��������������� gauss �����еĶ�Ӧλ��
        end
        respons = zeros(num, k); %���ڴ洢ÿ�����ݵ��ÿ����˹ģ�͵���Ӧ��
        
        total = sum(gauss, 2); %��gauss�����ÿһ�н�����Ͳ���
        for idx = 1:num
            respons(idx, :) = gauss(idx,:) ./ total(idx); %��ÿ�����ݵ��ÿ����˹ģ�͵���Ӧ�Ƚ��й�һ������./ ��ʾ��Ԫ�س�������
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
        [X,Y]=meshgrid(x,y); %��������x��y�а��������귵�ض�ά��������

        stepHandles = cell(1,k);
        ZTot = zeros(size(X));

        for idx = 1 : k
           Z = getGauss(mu(idx,:), sigma{idx}, X, Y); %getGauss�������ڼ����˹�ֲ��ĸ����ܶ�ֵ��
           Z = lambda(idx)*Z;
           [~,stepHandles{idx}] = contour(X,Y,Z); %ͨ��contour���������������ߣ�contour�������ڻ��Ƶȸ���ͼ��[~,stepHandles{idx}]��ʾֻ��ȡ�����ߵľ���������þ��������stepHandles����Ķ�Ӧλ�á�
           ZTot = ZTot + Z; %����ǰ��˹ģ�͵���Ӧ��Z�ӵ�ZTot����
        end
        hold off
       
        subplot(2,2,3) % image 3 - PDF for GMM
        mesh(X,Y,ZTot),title('PDF for GMM');%����һ������ͼ��������ͼΪ��ά����

        subplot(2,2,4) % image 4 - Projection of GMM
        surf(X,Y,ZTot),view(2),title('Projection of GMM')%GMMͶӰ�����ڻ�����ά����ͼ������ `ZTot` ��ֵ�������ϵ������Ⱦɫ�����ֳ���ά�������״����ɫ��`view(2)` ���ڽ�ͼ�ε��ӽ�����Ϊ�����Ϸ��۲�
        shading interp %shading interp �����ڸı����������ɫ��ʽ�ĺ�����ʹ�ò�ֵ�ķ�ʽ�����������ɫ��ʹ����ɫ�仯����ƽ������
        %colorbar����
        drawnow();%����ǿ��ˢ�²��ػ�ͼ�δ��ڵĺ�����ÿ�ε�����ʵʱ��ʾ���µĽ��
       
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
           for idx = 1 : k %forѭ�������ÿ����˹ģ�ͣ��������������ߵ�������ɫ����Ϊ����ʾ����
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