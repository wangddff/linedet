from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from src.database import get_db
from src.database.models import WiringRule

router = APIRouter()


class WiringRuleResponse(BaseModel):
    id: int
    product_id: int
    station_id: int
    layer: Optional[int]
    hole_number: str
    wire_number: Optional[str]
    wire_color: Optional[str]
    has_connector: bool
    has_short_wire: bool
    has_jumper: bool

    class Config:
        from_attributes = True


@router.get("/{product_id}", response_model=List[WiringRuleResponse])
async def get_wiring_rules(
    product_id: int, station_id: Optional[int] = None, db: Session = Depends(get_db)
):
    query = db.query(WiringRule).filter(WiringRule.product_id == product_id)
    if station_id:
        query = query.filter(WiringRule.station_id == station_id)

    rules = query.order_by(
        WiringRule.station_id, WiringRule.layer, WiringRule.hole_number
    ).all()
    return [
        WiringRuleResponse(
            id=r.id,
            product_id=r.product_id,
            station_id=r.station_id,
            layer=r.layer,
            hole_number=r.hole_number,
            wire_number=r.wire_number,
            wire_color=r.wire_color,
            has_connector=r.has_connector,
            has_short_wire=r.has_short_wire,
            has_jumper=r.has_jumper,
        )
        for r in rules
    ]


@router.post("/import")
async def import_wiring_rules(
    product_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    import pandas as pd
    import io

    content = await file.read()
    df = (
        pd.read_excel(io.BytesIO(content))
        if file.filename.endswith(".xlsx")
        else pd.read_csv(io.BytesIO(content))
    )

    required_columns = ["station_id", "hole_number"]
    for col in required_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"缺少必需列: {col}")

    db.query(WiringRule).filter(WiringRule.product_id == product_id).delete()

    rules = []
    for _, row in df.iterrows():
        rule = WiringRule(
            product_id=product_id,
            station_id=int(row.get("station_id", 1)),
            layer=int(row["layer"]) if pd.notna(row.get("layer")) else None,
            hole_number=str(row["hole_number"]),
            wire_number=str(row["wire_number"])
            if pd.notna(row.get("wire_number"))
            else None,
            wire_color=str(row["wire_color"])
            if pd.notna(row.get("wire_color"))
            else None,
            has_connector=bool(row.get("has_connector", False)),
            has_short_wire=bool(row.get("has_short_wire", False)),
            has_jumper=bool(row.get("has_jumper", False)),
        )
        rules.append(rule)

    db.add_all(rules)
    db.commit()

    return {"message": f"成功导入 {len(rules)} 条接线规则"}


@router.delete("/{product_id}")
async def delete_wiring_rules(
    product_id: int, station_id: Optional[int] = None, db: Session = Depends(get_db)
):
    query = db.query(WiringRule).filter(WiringRule.product_id == product_id)
    if station_id:
        query = query.filter(WiringRule.station_id == station_id)

    count = query.delete()
    db.commit()

    return {"message": f"删除 {count} 条规则"}
