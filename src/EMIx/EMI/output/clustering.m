clear

addpath('../../../scripts/npy-matlab-master/npy-matlab/')
addpath('../../../scripts')

% data = readNPY('Amat_1.0.npy'); 
data = readNPY('Amat_0.0001.npy'); 
A = create_sparse_mat_from_data(data);

N = length(A);

if N == 321
    N_in = 83;    
elseif N == 1153
    N_in = 291;
elseif N == 4353
    N_in = 1092;
elseif N == 16897
    N_in = 4226;
else
    disp('ERROR: N not supported...')
end

P = A;

for i = 1:N_in
  
    if sum(A(i,N_in:end)) ~= 0

        P(i,:) = 0;
        P(i,i) = 1;
    end 
end

for i = N_in:N
  
    if sum(A(i,1:N_in)) ~= 0

        P(i,:) = 0;
        P(i,i) = 1;
    end 
end

disp('Computed P')
%spy(P)

EPA = eig(full(P\A));
EA  = eig(full(A));

set_plot_defaults
plot(sort(real(EA)),'o')
hold on
plot(sort(real(EPA)),'o')
set_plot_defaults

epsilon = 0.1 * ones(size(EPA));

legend('$A$','$P^{-1}A$')

% semilogy(1 + epsilon)
% semilogy(1 - epsilon)

N_outliers = 0;

for i = 1:length(EPA)

    if abs(EPA(i) - 1) > epsilon

        N_outliers = N_outliers + 1;
    end
end

disp(N_outliers)
disp(N_outliers/N)





