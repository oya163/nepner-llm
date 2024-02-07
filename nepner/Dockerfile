# Pull the base image with python 3.9 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.9

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.9 -m pip install -r requirements.txt 

# Create the model directory for mounting the EFS 
RUN mkdir -p /mnt/ml

# Copy the earlier created app.py file to the container
COPY app.py ./

# Set the CMD to your handler
CMD ["app.lambda_handler"]

# ###########
# # BUILDER #
# ###########
# FROM public.ecr.aws/lambda/python:3.9 as builder

# # # Copy function code
# # COPY handler.py ${LAMBDA_TASK_ROOT} 
# # COPY requirements.txt  ${LAMBDA_TASK_ROOT} 

# # # install dependencies
# # RUN pip3 install --user -r requirements.txt

# # # Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# # CMD [ "handler.handle" ]

# RUN pip3 install --upgrade pip

# COPY requirements.txt  .
# RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# #########
# # FINAL #
# #########
# FROM public.ecr.aws/lambda/python:3.9
# RUN pip3 install --upgrade pip

# COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# COPY . ${LAMBDA_TASK_ROOT}

# CMD [ "handler.handle" ]