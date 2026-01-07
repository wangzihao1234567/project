clc;clear;close all;	
load('problem2_score.mat')	
test_data1=G_out_data.test_data1;	
zhibiao_label1=ones(1,size(test_data1,2));	
zhibiao_label=G_out_data.zhibiao_label;  %正向化指标设置	
if length(zhibiao_label)<length(zhibiao_label1)	
    zhibiao_label=[zhibiao_label,zhibiao_label1(length(zhibiao_label)+1:size(test_data1,2))];	
end	
A_data1=jisuan(test_data1,zhibiao_label);  %正向化之后的矩阵	
[n,~]=size(A_data1);	
A_data =A_data1 ./ repmat(sum(A_data1.*A_data1) .^ 0.5, n, 1); %矩阵归一化	
symbol_label=G_out_data.symbol_label;	
	
	
[Score_shangquan,quan_shang]=shangquanfa(A_data,symbol_label); 	
[Score_bianyi,quan_bianyi]=bianyixishu(A_data,symbol_label);	
W=[quan_shang;quan_bianyi];	
disp('现在正在使用  熵权-变异系数-博弈组合法')	
disp('熵权法得到权重：')	
disp(quan_shang)	
disp('变异系数法得到权重：')	
disp(quan_bianyi)	
Score_All=[Score_shangquan,Score_bianyi];	
	
for i=1:size(W,1)	
     for j=1:size(W,1)	
           W1(i,j)=W(i,:)*W(j,:)';	
      end	
     P(i,1)=W(i,:)*W(i,:)';	
 end 	
  A=(W1)^(-1)*P;	
  if min(A)<0 	
       A1=(A-min(A))/(sum((A-min(A))));	
   else	
    A1=A;	
   end 	
 W2=A1'*W;   Wc=W2/sum(W2);  %权重标准化	
	
	
	
disp(' 熵权-变异系数-博弈组合法组合得到的权重')	
disp(Wc)	
s=A_data*Wc';	
score=(100*s/max(s))';	
score_L=(mean(Score_All')-std(Score_All'));	
score_H=(mean(Score_All')+std(Score_All'));	
Out_table(:,1)=cell2table(G_out_data.table_str);	
Out_table(:,2)=array2table(G_out_data.table_data);	
Out_table(:,3)=array2table(G_out_data.score_L1);	
Out_table(:,4)=array2table(G_out_data.score_H1);	
Out_table.Properties.VariableNames={'评价对象','均分评分','评分下限','评分上限'};	
disp(Out_table)	
