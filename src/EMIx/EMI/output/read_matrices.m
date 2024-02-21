% addpath('../../../scripts/npy-matlab-master/npy-matlab/')
% addpath('../../../scripts')

N_samples = 11;
A    = cell(N_samples);
eigs_data = cell(N_samples);

data = readNPY('Amat_1.0.npy'); 
A{1} = create_sparse_mat_from_data(data);
data = readNPY('Amat_0.1.npy'); 
A{2} = create_sparse_mat_from_data(data);
data = readNPY('Amat_0.01.npy'); 
A{3} = create_sparse_mat_from_data(data);
data = readNPY('Amat_0.001.npy'); 
A{4} = create_sparse_mat_from_data(data);
data = readNPY('Amat_0.0001.npy'); 
A{5} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-05.npy'); 
A{6} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-06.npy'); 
A{7} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-07.npy'); 
A{8} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-08.npy'); 
A{9} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-09.npy'); 
A{10} = create_sparse_mat_from_data(data);
data = readNPY('Amat_1e-10.npy'); 
A{11} = create_sparse_mat_from_data(data);

rhs = readNPY('rhs.npy');

set_plot_defaults

conds = zeros(1,N_samples);
expected_cg_its = zeros(1,N_samples);
cg_its = zeros(1,N_samples);


its = 250;
resvec = zeros(1,its);

n_outliers = 32;

N = size(A{1},1);

cg_tol = 1e-5;
cg_maxit = 1000;

    f = ones(N,1)/norm(ones(N,1));
%f = rhs/norm(rhs);

% b = 7.8;

for i = 1:N_samples
    
    % make matrix symmetric 
    AA = ((A{i}-A{i}') == 0) .* A{i};
    
    %AA - AA'
           
    eigs_data{i} = sort(eig(full(AA))); 
    
    conds(i) = cond(full(AA));
    
    subplot(1,2,1)
    set_plot_defaults
    semilogy(eigs_data{i}, 'o')
    hold on
    if i == N_samples
    loglog(ones(1,N)*a,'black')
    loglog(ones(1,N)*b,'black')
    text(10,10,'$b$')
    text(0.1,100,'$a$')
    end
         
%     if i == 1
%      a = min(eigs_data{i}) - 0.0003;  
%     end
    
    a = min(eigs_data{1})-0.01;
    b = eigs_data{i}(N - n_outliers) + 0.1;
    sigma = (sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a));    
    %sigma = (sqrt(conds(i)) - 1)/(sqrt(conds(i)) + 1);    
    
%       for j = 1:its
        [x_cg,flag,relres,iter] = pcg(AA,f,cg_tol,cg_maxit);        
%         resvec(j) = relres;
%         if flag == 0
%             break
%         end 
% %       end
     x = AA\f;     
     err  = x-x_cg;
     err0 = x;
     epsilon = sqrt(err'*(AA*err))/sqrt(err0'*(AA*err0));
     expected_cg_its(i) = n_outliers + ceil(log(2/epsilon)/log(1/sigma));
     
     if flag ~= 0
         disp('CG not converged!')
     end
     cg_its(i) = iter;
     
%     subplot(1,3,3)
%     set_plot_defaults
%     semilogy(resvec)
%     ylabel('$\|\mathbf{r}\|_2$')
%     xlabel('iterations')
%     hold on
%     
     
end

legend('$\tau = 1$','$\tau = 10^{-1}$','$\tau = 10^{-2}$',...
    '$\tau = 10^{-3}$','$\tau = 10^{-4}$','$\tau = 10^{-5}$','$\tau = 10^{-6}$',...
    '$\tau = 10^{-7}$','$\tau = 10^{-8}$','$\tau = 10^{-9}$')
set_plot_defaults

subplot(1,2,2)
dts = 10.^-(0:N_samples-1);
set_plot_defaults
loglog(dts,conds,'o--')
hold on
loglog(dts,cg_its,'d-.')
loglog(dts,expected_cg_its,'s-')
set_plot_defaults
legend('$\kappa(A_n)$','CG iterations','CG iterations (theory)')
xlabel('$\tau$')




