% CS229, Autumn 2008-09
% Matlab Review Session

%%% elementary operations
5+6
3-2
5*8
1/2
2^6
1 == 2  % false
1 ~= 2  % true.  note, not "!="
1 && 0
1 || 0
xor(1,0)


%% variable assignment
a = 3; % semicolon suppresses output
b = 'hi';
c = 3>=1;

% Displaying them:
a = pi
disp(sprintf('2 decimals: %0.2f', a))
disp(sprintf('6 decimals: %0.6f', a))
format long
a
format short
a


%%  vectors and matrices
A = [1 2; 3 4; 5 6]

v = [1 2 3]
v = [1; 2; 3]
v = [1:0.1:2]  % from 1 to 2, with stepsize of 0.1. Useful for plot axes
v = 1:6   % from 1 to 6, assumes stepsize of 1

C = 2*ones(2,3)  % same as C = [2 2 2; 2 2 2]
w = ones(1,3)   % 1x3 vector of ones
w = zeros(1,3)
w = rand(1,3)  % drawn from a uniform distribution 
w = randn(1,3) % drawn from a normal distribution (mean=0, var=1)
w = -6 + sqrt(10)*(randn(1,10000))  % (mean = 1, var = 2)
hist(w)
e = []; % empty vector
I = eye(4)    % 4x4 identity matrix


%% dimensions
sz = size(A)
size(A,1)  % number of rows
size(A,2) % number of cols
length(v)  % size of longest dimension


%% loading data
load q1y.dat
load q1x.dat
who
whos
clear q1y  % clear w/ no argt clears all
v = q1x(1:10);
save hello v;   % save variable v into file hello.mat
save hello.txt v -ascii; % save as ascii
% fopen, fprintf, fscanf also work
% ls  %% cd, pwd  & other unix commands work in matlab; to access shell,
% preface with "!" 


%% indexing
A(3,2) % indexing is (row,col)
A(2,:)  % get the 2nd row. %% ":" means every elt along that dimension
A(:,2)  % get the 2nd col
A(1,end) % 1st row, last elt. Indexing starts from 1.
A(end,:) % last row
A([1 3],:)

A(:,2) = [10 11 12]'  % change second column
A = [A, [100; 101; 102]]; % append column vec
% A = [ones(size(A,1),1),  A];  % e.g bias term in linear regression
A(:) % Select all elements as a column vector.


%% matrix operations
A * C % matrix multiplication
B = [5 6; 7 8; 9 10] % same dims as A
A .* B % element-wise multiplcation
% A .* C  or A * B gives error - wrong dimensions
A .^ 2
1./v
log(v)  % functions like this operate element-wise on vecs or matrices 
exp(v) % e^4
abs(v)

-v  % -1*v

v + ones(1,length(v))
% v + 1  % same

A'  % (conjuate) transpose


%% misc useful functions

% max  (or min)
a = [1 15 2 0.5]
val = max(a)
[val,ind] = max(a)

% find
find(a < 3)
A = magic(3)
[r,c] = find(A>=7)

% sum, prod
sum(a)
prod(a)
floor(a) % or ceil(a)
max(rand(3),rand(3))
max(A,[],1)
min(A,[],2)
A = magic(9)
sum(A,1)
sum(A,2)
sum(sum( A .* eye(9) ))
sum(sum( A .* flipud(eye(9)) ))


% pseudo-inverse
pinv(A)  % inv(A'*A)*A'

% check empty
isempty(e)
numel(A)
size(A)
prod(size(A))


%% reshape and replication
A = magic(3)
A = [A [0;1;2]]
reshape(A,[4 3])
reshape(A,[2 6])
v = [100;0;0]
A+v
A + repmat(v,[1 4])
bsxfun(@plus, A, v)


%% plotting
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t); 
plot(t,y1);
y2 = cos(2*pi*4*t);
hold on;  % "hold off" to turn off
plot(t,y2,'r--');
xlabel('time');
ylabel('value');
legend('sin','cos');
title('my plot');
close;  % or,  "close all" to close all figs

figure(2), clf;  % can specify the figure number
subplot(1,2,1); % Divide plot into 1x2 grid, access 1st element
plot(t,y1);
subplot(1,2,2); % Divide plot into 1x2 grid, access 2nd element
plot(t,y2);
axis([0.5 1 -1 1]);  % change axis scale

%% display a matrix (or image) 
figure;
imagesc(magic(15)), colorbar, colormap flag;
% comma-chaining function calls.


%% for, while, if statements

w = [];
z = 0;
is = 1:10
for i=is
    w = [w, 2*i] % Same as \/
%     w(i) = 2*i
%     w(end+1) = 2*i
    z = z + i;
    break;
    continue;
end
% avoid! same as w = 2*[1:10], z = sum([1:10]);

w = [];
while length(w) < 3
    w = [w, 4];
    break    
end


if w(1)==0
    % <statement>
elseif w(1)==1   
    % <statement>
else
    % <statement>
end


% exit  % quit

%% Logistic regression

[th,ll] = logistic_grad_ascent(q1x,q1y);
plot(ll)
