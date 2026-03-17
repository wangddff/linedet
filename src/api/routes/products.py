from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from src.database import get_db
from src.database.models import Product

router = APIRouter()


class ProductCreate(BaseModel):
    name: str
    code: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class ProductResponse(BaseModel):
    id: int
    name: str
    code: str
    description: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[ProductResponse])
async def list_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    return [
        ProductResponse(
            id=p.id,
            name=p.name,
            code=p.code,
            description=p.description,
            created_at=p.created_at.isoformat() if p.created_at else "",
        )
        for p in products
    ]


@router.post("", response_model=ProductResponse)
async def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    existing = db.query(Product).filter(Product.code == product.code).first()
    if existing:
        raise HTTPException(status_code=400, detail="产品编号已存在")

    db_product = Product(
        name=product.name, code=product.code, description=product.description
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    return ProductResponse(
        id=db_product.id,
        name=db_product.name,
        code=db_product.code,
        description=db_product.description,
        created_at=db_product.created_at.isoformat() if db_product.created_at else "",
    )


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="产品不存在")

    return ProductResponse(
        id=product.id,
        name=product.name,
        code=product.code,
        description=product.description,
        created_at=product.created_at.isoformat() if product.created_at else "",
    )


@router.delete("/{product_id}")
async def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="产品不存在")

    db.delete(product)
    db.commit()
    return {"message": "删除成功"}
