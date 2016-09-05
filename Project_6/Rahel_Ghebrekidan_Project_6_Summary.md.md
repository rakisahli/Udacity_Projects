
### Project 6: Make Effective Data Visualization
### Udacity Nanodegree Data Analyst 
### Rahel Ghebrekidan
### September 1, 2016


# Introduction 

In project 2 (Investigating dataset) I have selected Titanic dataset and analyzed the effect of demographic and socio-economic factors on  survival of passengers. I analyzed Age as one of the factors. I displayed the relation between the age (age group) and proportion of survived passengers using bar graph. In this project, I want to do similar graph using D3.js to show the percentage of survived passengers in each age group. 
The graph and table from project 2 are presented here below. 


![](https://raw.githubusercontent.com/rakisahli/Udacity_Projects/master/Project_6/Table_from_P2.JPG)
![](https://raw.githubusercontent.com/rakisahli/Udacity_Projects/master/Project_6/Bar_Chart_from_P2.JPG)

# Data and Visualization selection 

I have chosen to use bar graph because bar graph is good to compare different groups and my independent variable is categorical data. In this graph, the x- axis (independent variable) is age group binned with 10 year’s intervals. I have used binned age group instead of using simply age because using age group is easy to explain and visualize than using just the age. At first I was thinking of grouping the age as less than 18 years and great than 18 years. The relation between age and survived passengers will be more clear when there are more groups . So I decided to use age groups binned with ten year’s interval because in most stages of age, people within ten years of age interval have similar physical appearance. In addition, the number of age groups, I will have when using 10 years of interval looks good when displayed than when having greater than eight groups or less groups.
My y- axis (dependent variable) is survived passengers in percentage. The aim of the project is to see the effect of age on survival or to show how the survival rate was higher among younger passengers. I have used the survived percentage to perished percentage. It could be also done the other way, “how the perished rate was lower among the younger passengers but I prefer to use the positive term. So I have used survived percentage rather than perished percentage.  

The data I used is taken from the above table and saved in CVS format. Age grouping and aggregating the data was already done in project 2. [click here](https://github.com/rakisahli/Udacity_Projects/tree/master/Project_2)


# My First Draft Bar Chart

![My First Draft Bar chart](https://raw.githubusercontent.com/rakisahli/Udacity_Projects/master/Project_6/Bar_Chart_ver01.JPG)

# Feedback on first graph

Then I have shared the graph with my three friends and I have got the following main feedbacks. The main feedbacks were:
1.	To put values (percentage of survived passengers) of each age group on the respective bars
2.	To add "in years” to the x- axis label “Age Group”) as it might be confusing if the age groups are in months or years
3.	To add hover to make it look great 
4.	To put date and my name
          
In addition to the feedbacks I got, I have also made minor changes on the x-axis line and background of the graph. I have changed the title of the graph after I got comments from two Udacity Reviewer. The final graph is displayed below


![](https://raw.githubusercontent.com/rakisahli/Udacity_Projects/6ec2bab25a2b26d4ef3503d33a5e694afab892b9/Project_6/Bar_Chart_final.JPG)


# Reference:

1.	Scott Murray(2016). Making a bar chart: Retrieved from  http://alignedleft.com/tutorials/d3/making-a-bar-chart 
2.	Mbostock(2016). D3 API Reference: Retrieved from https://github.com/d3/d3/blob/master/API.md#collections- 

