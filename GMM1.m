% Initialization
clc;
clear;
clf;

% data 1
mu1 = [0 0];    sigma1 = [2 -0.9; -0.9 2]; 
r1 = mvnrnd(mu1,sigma1, 100);
% data 2
mu2 = [5 3];    sigma2 = [3 2; 2 2];
r2 = mvnrnd(mu2,sigma2, 100);

figure(1)
for i = 1:2
    subplot(2,2,i);
    title('Original data');
    plot(r1(:,1),r1(:,2),'r+'); 
    hold on; 
    plot(r2(:,1),r2(:,2),'b+');
    hold off; 
end
data = [r1; r2]; 
fit_gmm(data,2,0.1);

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

function [lambda, mu, sigma] = fit_gmm(data, k, precision)
    [num,dim] = size(data);  
    lambda = repmat(1/k,k,1); 
    
    randIdx = randperm(num);   
    mu = data(randIdx(1:k),:); 
    
    dataVariance =  cov(data,1);   
    sigma = cell (1, k);  
    
    for i = 1 : k
        sigma{i} = dataVariance;
    end
    
    x=-5:0.05:10;
    y=-5:0.05:10;
    
    iter = 0; precious_L = 100000;
    while iter < 100
        % E-step
        gauss = zeros(num, k); 
        for idx = 1: k
            gauss(:,idx) = lambda(idx)*mvnpdf(data, mu(idx,:), sigma{idx});
        end
        respons = zeros(num, k); 
        
        total = sum(gauss, 2); 
        for idx = 1:num
            respons(idx, :) = gauss(idx,:) ./ total(idx); 
        end

       % M-step
       responsSumedRow = sum(respons,1);
       responsSumedAll = sum(responsSumedRow,2);
       for i = 1 : k
          lambda(i) =  responsSumedRow(i) / responsSumedAll;
          newMu = zeros(1, dim);
          for j = 1 : num
              newMu = newMu + respons(j,i) * data(j,:);
          end
          mu(i,:) = newMu ./ responsSumedRow(i);
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
        [X,Y]=meshgrid(x,y);

        stepHandles = cell(1,k);
        ZTot = zeros(size(X));

        for idx = 1 : k
           Z = getGauss(mu(idx,:), sigma{idx}, X, Y); 
           Z = lambda(idx)*Z;
           [~,stepHandles{idx}] = contour(X,Y,Z); 
           ZTot = ZTot + Z;
        end
        hold off
       
        subplot(2,2,3) 
        mesh(X,Y,ZTot),title('PDF for GMM');

        subplot(2,2,4) 
        surf(X,Y,ZTot),view(2),title('Projection of GMM')
        shading interp 
        drawnow();

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
           for idx = 1 : k 
                set(stepHandles{idx},'LineColor','none')
           end
       end
       precious_L = L;
    end
end 