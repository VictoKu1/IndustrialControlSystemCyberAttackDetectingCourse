# Docker container for running a github reposiory
# The system is an AI detection for an industrial power control system,
# which allows the user to add a csv file and recieve a classification
# of the data, if it is a regular operation, natural event or a cyber attack
# using a python script.





FROM python:3.8



# Install dependencies
RUN pip install --upgrade pip
RUN pip install pickle-mixin numpy pandas scikit-learn
RUN pip install --upgrade pickle-mixin numpy pandas scikit-learn


# Copy the necessary files from the host machine to the container
COPY Industrial_power_control_system_cyber_attacks_detetection_ui.py .

#Download the github repository
RUN git clone https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse

# Set the working directory and pull the latest version of the repository
WORKDIR /IndustrialControlSystemCyberAttackDetectingCourse
RUN git pull

# Set the command to run when the container starts
CMD ["python", "Industrial_power_control_system_cyber_attacks_detetection_ui.py"]

