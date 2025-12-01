from fastapi import FastAPI ,Path ,HTTPException ,Query
import json
import os

app= FastAPI()

def load_data():
    with open("patients.json", "r") as f:
        return json.load(f)

@app.get("/")
def hello():
    return {"message": "Patient management system api!"}

@app.get("/about")
def about():
    return {"message": "a fully functional api to manage patient records."}

@app.get("/view")
def view():
    data = load_data()
    return data


#PATH PARAMETRS
@app.get("/patient/{patient_id}")
def view_patient(patient_id:str = Path(...,description="The ID of the patient to retrieve from the DB",example="P001" )):

    # Load data
    data = load_data()
    if patient_id in data:
        return data[patient_id] #returns associated data if patient_id found
    raise HTTPException(status_code=404,detail="Patient ID not found")



# QUERY parameters 
@app.get("/sort")
def sort_patients(sort_by:str = Query(...,descrption="sort on basis of height,weight and bmi"),order:str=Query("asc",descrption="sort in ascenfing and descending order" )):

    valid_fields = ["height","weight","bmi"]
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400,detail=f"Invalid sort_by field. Must be one of {valid_fields}")
    
    if order not in ["asc","desc"]:
        raise HTTPException(status_code=400,detail="Invalid order value. Must be 'asc' or 'desc'")
    

    data=load_data()

    sort_order= True if order=="desc" else False

    sorted_data= sorted(data.values(), key=lambda x: x.get(sort_by,0), reverse=sort_order)

    return sorted_data
