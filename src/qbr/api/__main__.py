"""Start the development server with: python -m qbr.api"""
import os
import uvicorn

def main() -> None:
    uvicorn.run('qbr.api.main:app', host=os.getenv('QBR_HOST', '127.0.0.1'),
                port=int(os.getenv('QBR_PORT', '8000')), reload=os.getenv('QBR_RELOAD', '1') == '1')

if __name__ == '__main__':
    main()
