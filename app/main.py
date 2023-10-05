from fastapi import FastAPI, status

from app.apis import v1_router


app = FastAPI(title='Toxisity checker',
              description='Fastapi service for check toxisity',
              version='0.1')

# Adding v1 namespace route
app.include_router(v1_router)
print('router add succeed')

print('Пидор')

@app.get('/health',
         tags=['System probs'])
def health() -> int:
    return status.HTTP_200_OK
