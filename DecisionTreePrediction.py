'''
Importing modules:
-DecisionTreeClassifier (DTC) :-sklearn.tree
-export_graphviz :-sklearn.tree
-Image :-IPython.display
-pydotplus (pdp) 
-pandas (pd)
-accuracy_score (a_s) :-sklearn.metrics
-train_test_split (tts) :-sklearn.model_selection
-StringIO :-sklearn.externals.six
-time (tm4)
'''
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus as pdp
import pandas as pd
from sklearn.metrics import accuracy_score as a_s
from sklearn.model_selection import train_test_split as tts
from sklearn.externals.six import StringIO
import time as tm


#Defining a function which prints a message acknowledging the input provided
def PrintAcceptanceMessage():
  print("Request Accepted")

#Defining a function to enable the user to provide a custom pparameter for the depth of the tree plot
def ReturnMaxDepthValue():
    max_depth_input_param=float(input("Enter max depth:"))

    #Verifying whether  the provided input is greater than 7 or not
    #Case-1
    if(max_depth_input_param>7):
      print("The value cannot exceed 7.")
      print("Setting value to 7...")
      return 7
    #Case-2  
    else:
      return max_depth_input_param  




#Defining a function to create decision tree plot from the given data
def CreateDecisionTreePlot(factors_arg,result_arg,list_arg):
  factors_train_param,factors_test_param,result_train_param,result_test_param=tts(factors_arg,result_arg,train_size=0.75,random_state=0)

  global_func_max_depth_param=None
  CLF_param=None

  specify_max_depth_input_param=input("Specify the max depth value?(:-Yes or No)")

  #Assessing the user's option to provide a custom value to the maximum depth
  #Case-1
  if(specify_max_depth_input_param=="Yes" or specify_max_depth_input_param=="yes"):
    global_func_max_depth_param=ReturnMaxDepthValue()
    CLF_param=DTC(max_depth=global_func_max_depth_param)
  #Case-2  
  else:
    #Printing the ending message
    PrintAcceptanceMessage()
    CLF_param=DTC()


  CLF_param.fit(factors_train_param,result_train_param)

  prediction_y_param=CLF_param.predict(factors_test_param)


  accuracy_param=a_s(result_test_param,prediction_y_param)*100
  print("The veracity of the data is {}%".format(round(accuracy_param,2)))

  StringDF_param=StringIO()

  export_graphviz(CLF_param,out_file=StringDF_param,filled=True,rounded=True,special_characters=True,feature_names=list_arg,class_names=["0","1","2","3","4","5","6"])

  graph_param=pdp.graph_from_dot_data(StringDF_param.getvalue())

  graph_param.write_png("Survival Rates")

  image_param=Image(graph_param.create_png())

  return image_param


#Defining the main function
def Main():
  print("Welcome to SurvivalPredictorTitanic.py")

  view_information_titanic=input("Don't know about the Titanic?(:-I Know,I Don't Know)")
  #Verifying the user's choice whether they have pre-requisiste knowledge of Titanic
  #Case-1
  if(view_information_titanic=="I Don't Know" or view_information_titanic=="i don't know" or view_information_titanic=="I don't know" or view_information_titanic=="I Don't know" or view_information_titanic=="I don't Know"):
    print("The Titanic was a British passenger ocean liner, created by then prominent ship-building enterprise, White Star Line.") 
    tm.sleep(2.3)
    print("It was built in Belfast and was supposed to conduct its maiden voyage in 10 April 1912, from Southampton to New York.")
    tm.sleep(2.4)
    print("However, on the night of April 15, midway through its voyage, it struck an iceberg.")
    tm.sleep(3.3)
    print("The damage was fatal, the ship sank within two and a half hours.")
    tm.sleep(2.3)
    print("Holes were ruptured in the anterior of the ship, causing the bow to be englufed in water first, drasitcally increasing the steepness of its angle.")
    tm.sleep(3.4)
    print("Two hours after the collision, the ship was practically upright in the water. Enormous pressure generated on the stern caused the massive machine  to rip in half.")
    tm.sleep(2.5)
    print("500 people were rescued, while 1,500 perished.")
    tm.sleep(1.2)
    print("A majority of the people died due to a lack of lifeboats.")
    tm.sleep(2.9)
    print("The Titanic was considered unsinkable, hence the idea of equipping the Titanic with sufficient lifeboatswas considered obsolete")
    tm.sleep(4.0)
    print("To know more about the Titanic, visit:'https://en.wikipedia.org/wiki/Titanic#Maiden_voyage'")
  #Case-2
  else:
    #Printing the ending message
    PrintAcceptanceMessage()

  view_information_tree_plot=input("Don't know what a Decision Tree Plot is?(:-I Know,I Don't Know)")
  #Verifying the user's choice whether they have pre-requisiste knowledge of a Dcision Tree
  #Case-1
  if(view_information_tree_plot=="I Don't Know" or view_information_tree_plot=="i don't know" or view_information_tree_plot=="I don't know" or view_information_tree_plot=="I Don't know" or view_information_tree_plot=="I don't Know"):
    print("A Decision Tree Plot is a hiearchical flowchart, which has branches based on a binary value or an expression.") 
    tm.sleep(2.3)
    print("One branch depicts the resultant actions, if the value or expression is true")
    tm.sleep(2.4)
    print("Another branch depicts the scenario of the value being false.")
    tm.sleep(3.3)
    print("Decision Tree Plots can be used to depict the result of logistic regression and multilinear logistic regression in a more vivid manner.")
    tm.sleep(2.3)
    print("However,the structure of the dcision tree is soleley dependent on the logisitc regression or multilinear logistic regression model, which also refers to its accuracy.")
    tm.sleep(1.2)
    print("Hence, Decision Tree Plots are not always precise.")
    tm.sleep(2.3)
    print("Decision Tree Plots are used in several areas such as:")
    tm.sleep(1.9)
    print("1. To study the relation of how several values in a dataset are correlated to each other and how they affect the final value and their co-contributors")
    tm.sleep(2.8)
    print("2. To provide an enhanced summarization of the regression model, its accuracy and results.")
    tm.sleep(3.4)
    print("These uses are incorporated in almost all fields")
    tm.sleep(3.4)
    print("To know more about the Titanic, visit:'https://en.wikipedia.org/wiki/Decision_tree'")
  #Case-2
  else:
    #Printing the ending message
    PrintAcceptanceMessage()
  


  #Assigning column names to the dataset
  df_col=["PassengerClass","Male","Age","Siblings/Spouse","Parent/Child","Survived"]
  
  #Reading data from the file
  df=pd.read_csv("data.csv",names=df_col).iloc[1:]

  result_list=["Unusable_Element","Male","Siblings/Spouse","Parent/Child","Survived"]
  result_count=0

  for result in result_list[1:]:
    result_count+=1
    print("{}:{}".format(result_count,result))

  result_input=int(input("Enter number:"))
  result_choice=result_list[result_input]

  result_remove_list=["PassengerClass","Male","Age","Siblings/Spouse","Parent/Child","Survived"]
  result_remove_list.remove(result_choice)

  factors=df[result_remove_list]
  result=df[result_choice]


  image_main_param=CreateDecisionTreePlot(factors,result,result_remove_list)
  
  print("Thank You for using SurvivalPredictor.py")

  return image_main_param

Main()
 


    

