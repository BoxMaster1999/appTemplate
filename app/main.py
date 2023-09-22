from fastapi import FastAPI, status

from app.apis import v1_router


app = FastAPI(title='appTemplate',
              description='Fastapi template',
              version='0.1')

# Adding v1 namespace route
app.include_router(v1_router)


@app.get('/health',
         tags=['System probs'])
def health() -> int:
    return status.HTTP_200_OK
