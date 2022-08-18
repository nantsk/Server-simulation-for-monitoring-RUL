# Server-simulation-for-monitoring-RUL
Server simulation for monitoring remaining useful life of turbofan jet engines

To demonstrate the working of proposed models in a real-time environment of https://github.com/ritu-thombre99/RUL-Prediction, we
developed a website to simulate the server that can monitor the health of the engines as
follows:
+ After the simulation is started, health scores of 20 engines are displayed in a table
+ Health of an engine falls under four categories:
  + Extremely Critical (0-0.25)
  + Critical (0.25-0.5)
  + Normal (0.5-0.75)
  + Optimal (0.75-1)
+ Statistics of health and range of health scores of engines is displayed as a pie
chart and a line graph
+ Health scores of engines are refreshed every 1-2 second
