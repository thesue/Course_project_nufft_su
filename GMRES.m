%基本的GMRES算法

function [x,n_iteration,res] = GMRES(A,b,x0,Max_iteration,eps)
r0 = b-A*x0;
beta = norm(r0,2);
n = size(A,1);
v = zeros(n,n);
H = zeros(n,n);
v(:,1) = r0/beta;
m = length(x0);
res_plot = zeros(m,1);
for j = 1:Max_iteration
    n_iteration = j;
    w = A*v(:,j);
    for i = 1:j
        H(i,j) = v(:,i)'*w;
        w = w-H(i,j)*v(:,i);
    end
    H(j+1,j) = norm(w,2);
    if H(j+1,j) == 0
        break
    else
        v(:,j+1) = w/H(j+1,j);
    end
    barH = zeros(j+1,j);
    barH(1:j,1:j) = H(1:j,1:j);
    barH(j+1,j) = H(j+1,j);
    [Q,R] = qr(barH);
    g = Q'*(beta*eye(j+1,1));
    y = R\g;
    x = x0+v(:,1:j)*y;
    res =abs(g(end));
    res_plot(j,1)=res;
    if res < eps
        break
    end
end
%hold on 
%plot(log(res_plot))
end 