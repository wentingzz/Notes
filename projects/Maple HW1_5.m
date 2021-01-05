col1 = csvread('seeds.csv',1,0,[1,0,210, 0])
col2 = csvread('seeds.csv',1,4,[1,4,210, 4])
A = [col1, col2]
n = normc(A)
std = zscore(A)
range(n)
range(std)

plot(A(:,1,:),A(:,2,:))
plot(n(:,1,:),n(:,2,:))
plot(std(:,1,:),std(:,2,:))

P_x = mean(A(:,1,:))
P_y = mean(A(:,2,:))
