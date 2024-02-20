function A = create_sparse_mat_from_data(data)


rows = data(:,1) + 1;
cols = data(:,2) + 1;
vals = data(:,3);

A = sparse(rows, cols, vals);

end

