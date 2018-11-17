function [group,counter] = MinIndex(x,index)
   group= zeros(2,3);

   for i=1:max(index)

        counter(i) = 1;
   end
   for w=1:length(x)

       k = abs(((index(w))*3)-2);
       j = (index(w))*3;
       group(counter(index(w)),k:j) = x(w,:);
       counter(index(w)) = counter(index(w)) + 1;

   end 
    
end
