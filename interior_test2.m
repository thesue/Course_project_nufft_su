
% interior Dirichlet boundary condition
% boundary parameter equation (x=f(t) ,y=g(t))
% boundary condition  f(x,y) on B 
format long

N=100;
M=16*N;

N1=160;
M1=10*N1;

%target point in D
tar=[g1(0);g2(0)];

%initialize
h=2.0*pi/N;
h1=2.0*pi/N1;
t=zeros(N+1,1);
t1=zeros(N1+1,1);
A=zeros(M,M);
I=zeros(M,M);
b=zeros(M,1);
x0=zeros(M1,1);

%indentity matrix
for i=1:M
    I(i,i)=1.0;
end

%Gauss-legendre points
t(1)=0.0;
for i=1:N
    t(i+1)=h*i;
end

t1(1)=0.0;
for i=1:N1
    t1(i+1)=h1*i;
end



theta=zeros(M,1);
weights=zeros(M,1);


theta1=zeros(M1,1);
weights1=zeros(M1,1);

for i=1:N
    [x,w]=legpts(16,[t(i),t(i+1)]);
    theta((i-1)*16+1:i*16,1)=x;
    weights((i-1)*16+1:i*16,1)=w;
end


for i=1:N1
    [x1,w1]=legpts(10,[t1(i),t1(i+1)]);
    theta1((i-1)*10+1:i*10,1)=x1;
    weights1((i-1)*10+1:i*10,1)=w1;
end



for i=1:M1
    %target point
    x=zeros(2,1);
    x(1)=g1(theta1(i));
    x(2)=g2(theta1(i));
    for j=1:M
        % source points
        y=zeros(2,1);
        ny=zeros(2,1);
        y(1)=g1(theta(j));
        y(2)=g2(theta(j));
        ny(2)=-g1_gradient(theta(j));
        ny(1)=g2_gradient(theta(j));
        A(i,j)=Kfun(x,y,ny)*weights(j);
    end
    b(i)=f(x(1),x(2));
end

% GMRES solving linear system equation
[mu,n_interation,res]=GMRES(0.5*I-A,b,x0,9,1e-6);

%compute double-layer potential
sum=0;
for i=1:M1
    y=zeros(2,1);
    ny=zeros(2,1);
    y(1)=g1(theta1(i));
    y(2)=g2(theta1(i));
    ny(2)=-g1_gradient(theta1(i));
    ny(1)=g2_gradient(theta1(i));
    sum=sum+Kfun(tar,y,ny)*weights1(i)*mu(i);
end





% double layer kernel
function K = Kfun(x,y,ny)
  dx = x(1) - y(1);
  dy = x(2) - y(2);
  dr = sqrt(dx^2 + dy^2);

  rdotn = dx*ny(1) + dy*ny(2);
  K = 1/(2*pi)*rdotn/dr^2;
end

function x=g1(t)
  r=0.25+0.01*cos(6*t)+0.01*sin(3*t)+0.02*cos(5*t)+0.01*cos(8*t)+0.01*cos(10*t);
  x=r*cos(t);
end



function y=g2(t)
  r=0.25+0.01*cos(6*t)+0.01*sin(3*t)+0.02*cos(5*t)+0.01*cos(8*t)+0.01*cos(10*t);
  y=r*sin(t);
end


function value=f(~,~)
  value=1.0;
end

function x=g1_gradient(t)
  r_grad=-0.06*sin(6*t)+0.03*cos(3*t)-0.1*sin(5*t)-0.08*sin(8*t)-0.1*sin(10*t);
  r=0.25+0.01*cos(6*t)+0.01*sin(3*t)+0.02*cos(5*t)+0.01*cos(8*t)+0.01*cos(10*t);
  x=r_grad*cos(t)-r*sin(t);
end

function y=g2_gradient(t)
  r_grad=-0.06*sin(6*t)+0.03*cos(3*t)-0.1*sin(5*t)-0.08*sin(8*t)-0.1*sin(10*t);
  r=0.25+0.01*cos(6*t)+0.01*sin(3*t)+0.02*cos(5*t)+0.01*cos(8*t)+0.01*cos(10*t);
  y=r_grad*sin(t)+r*cos(t);
end