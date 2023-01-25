## Steps to Setting up the project on local machine (Windows) to run the project


#### Note: Make sure Python has installed on your machine. If not, get it from here https://www.python.org/downloads/


1) Open Windows Command Prompt and run the following command to clone the project to your local machine.

   #### git clone https://github.com/dcfe-tool/dcfetool.git
   
   
2) Open the project via any IDE tool. (Example: Visual Studio Code. download it here https://code.visualstudio.com/Download)



3) Open a New Terminal Window (command propt) from IDE Tools Menubar or Use the Shortcut Ctrl+Shift+`


   ![image](https://user-images.githubusercontent.com/123196611/214689147-ed9cc9bf-2f05-484a-b47f-c8964fcd9959.png)
   
   
  
4) Run the follwing command in the Terminal window.



   #### --> cd ./dcfetool                     (to change the path to the directory which contains the Manage.py file)
   
   #### --> pip install Django                (to install the Django package)
   
   #### --> pip install -r requirements.txt   (to install all the neccessary packages)
   
   
   
5) Execute the following command to run the application.



   #### --> python manage.py runserver


6) You will get something as showned in the image below. Click the url http://127.0.0.1:8000 and the application will be rendered on browser.



   ![image](https://user-images.githubusercontent.com/123196611/214686873-a997dfa6-a3e6-4468-9c37-fb1b6acf64f5.png)
