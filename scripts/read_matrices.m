% addpath('../../../../scripts/npy-matlab-master/npy-matlab/')
% addpath('../../../../scripts')

data = readNPY('Amat.npy');
A = create_sparse_mat_from_data(data);

data = readNPY('Pmat.npy');
P = create_sparse_mat_from_data(data);

compute_spectrum = true;
solve_flag = false;

if compute_spectrum

    [UA,SA,VA] = svd(full(A));
    [UP,SP,VP] = svd(full(P));
    PA = P\A;
    [UPA,SPA,VPA] = svd(full(PA));

    figure
    subplot(1,2,1)
    semilogy(diag(SA),'o')
    set_plot_defaults
    hold on
    semilogy(diag(SP),'.')
    semilogy(diag(SPA),'+')
    legend('$\sigma(A)$','$\sigma(P)$','$\sigma(P^{-1}A)$')

    EA = eig(full(A));
    EP = eig(full(P));
    EPA = eig(full(PA));

    subplot(1,2,2)
    semilogy(sort(real(EA)),'o')
    set_plot_defaults
    hold on
    semilogy(sort(real(EP)),'.')
    semilogy(sort(real(EPA)),'+')
    legend('$\lambda(A)$','$\lambda(P)$','$\lambda(P^{-1}A)$')
end

if solve_flag

    tol = 1e-10;

    % b = ones(length(A),1);
    bvec;
    
    disp('Solve with P')
    [x,flag,relres,iter,resvec] = gmres(A,b',[],tol,1000,P);
    disp(iter(2))
    disp(relres)
    semilogy(resvec,'-o')

    disp('Solve with P block diagonal')    
    [x,flag,relres,iter,resvec] = gmres(A,b',[],tol,1000,P_BD);
    disp(iter(2))
    disp(relres)

    hold on
    semilogy(resvec,'-o')
    legend('P','P BD')

end


