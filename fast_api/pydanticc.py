# from pydantic import BaseModel,EmailStr,AnyUrl,Field
# from typing import List ,Dict ,Optional,Annotated

# class Patient(BaseModel):
#     # name:str=Field(min_length=2 ,max_length=50)
#     name:Annotated[str,Field(max_length=50,title="Full Name of the patient",description="Name should be between 2 to 50 characters",example="John Doe")]
#     age:int =Field(gt=0,lt=100)
#     email:EmailStr
#     website:AnyUrl
#     weight:float=Field(gt=0,strict=True) #to avoid coercion
#     allergies:Optional[List[str]]=None
#     contacs:Dict[str,str] #this is a 2 level validation first to validate its dict and then key and value both are validated as str

# def insert_patient_data(patient:Patient):
#     print(patient.name)
#     print(patient.age)
#     print("Patient data inserted successfully.")



# patient_info={"name":"John Doe","age":30}
# patient1=Patient(**patient_info)
# insert_patient_data(patient1)







#FIELD VALIDATIONS

# from pydantic import BaseModel,EmailStr,AnyUrl,Field,FieldValidator
# from typing import List ,Dict ,Optional,Annotated

# class Patient(BaseModel):
#     # name:str=Field(min_length=2 ,max_length=50)
#     name:str
#     age:int
#     email:EmailStr
#     website:AnyUrl
#     weight:float=Field(gt=0,strict=True) #to avoid coercion
#     allergies:Optional[List[str]]=None
#     contacs:Dict[str,str] #this is a 2 level validation first to validate its dict and then key and value both are validated as str

#     @field_validator ("email",mode="after") #basically before and after means before and after pydantic's own validation
#     @classmethod
#     def email_validator(cls,value):

#         valid_domains=["gmail.com","yahoo.com","hotmail.com"]
#         domain_name=value.split("@")[-1]

#         if domain_name not in valid_domains:
#             raise ValueError(f"Email domain must be one of {valid_domains}")
#         return value


# def insert_patient_data(patient:Patient):
#     print(patient.name)
#     print(patient.age)
#     print("Patient data inserted successfully.")



# patient_info={"name":"John Doe","age":30}
# patient1=Patient(**patient_info)
# insert_patient_data(patient1)


#Model validators

# from pydantic import BaseModel,EmailStr,AnyUrl,Field,FieldValidator,ModelValidator
# from typing import List ,Dict ,Optional,Annotated

# class Patient(BaseModel):
#     # name:str=Field(min_length=2 ,max_length=50)
#     name:str
#     age:int
#     email:EmailStr
#     website:AnyUrl
#     weight:float=Field(gt=0,strict=True) #to avoid coercion
#     allergies:Optional[List[str]]=None
#     contacs:Dict[str,str] #this is a 2 level validation first to validate its dict and then key and value both are validated as str

#     @model_validator(mode="after")
#     def validate_emergency_contact(cls,model):
#         if model.age >60:
#             if "emergency" not in model.contacs:
#                 raise ValueError("Emergency contact is required for patients above 60 years old")


# def insert_patient_data(patient:Patient):
#     print(patient.name)
#     print(patient.age)
#     print("Patient data inserted successfully.")



# patient_info={"name":"John Doe","age":30}
# patient1=Patient(**patient_info)
# insert_patient_data(patient1)



#Computed Fields

# from pydantic import BaseModel,EmailStr,AnyUrl,Field,FieldValidator
# from typing import List ,Dict ,Optional,Annotated

# class Patient(BaseModel):
#     # name:str=Field(min_length=2 ,max_length=50)
#     name:str
#     age:int
#     email:EmailStr
#     website:AnyUrl
#     height:float=Field(gt=0,strict=True)
#     weight:float=Field(gt=0,strict=True) #to avoid coercion
#     allergies:Optional[List[str]]=None
#     contacs:Dict[str,str] #this is a 2 level validation first to validate its dict and then key and value both are validated as str

#     @computed_field
#     @property
#     def bmi(self)->float:  
#         return round(self.weight/(self.height**2),2)
    
    


    

# def insert_patient_data(patient:Patient):
#     print(patient.name)
#     print(patient.age)
#     print("Patient data inserted successfully.")



# patient_info={"name":"John Doe","age":30}
# patient1=Patient(**patient_info)
# insert_patient_data(patient1)


#Nested models

# from pydantic import BaseModel

# class Address(BaseModel):
#     city:str
#     state:str
#     pin:str

# class Patient(BaseModel):
#     name:str
#     gender:str
#     age:int
#     address:Address

# address_dict={'city':"mumbai",'state':"maharashtra",'pin':"400001"}
# address1=Address(**address_dict)
# patient_dict={"name":"alex","gender":"male","age":29,'address':address1}
# patient1=Patient(**patient_dict)
# print(patient1)


#serializaiton

from fast_api.pydanticc import BaseModel

class Address(BaseModel):
    city:str
    state:str
    pin:str

class Patient(BaseModel):
    name:str
    gender:str
    age:int
    address:Address

address_dict={'city':"mumbai",'state':"maharashtra",'pin':"400001"}
address1=Address(**address_dict)
patient_dict={"name":"alex","gender":"male","age":29,'address':address1}
patient1=Patient(**patient_dict)

# patient1.model_dump_json(include=["name","gender"]) #gives json string and an dictionary as well
patient1.model_dump_json(exclude_unset=True) #does not include fields which were not set during initialization

