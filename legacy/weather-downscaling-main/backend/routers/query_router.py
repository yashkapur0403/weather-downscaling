from fastapi import APIRouter, Depends
from backend.controllers.query_controller import handle_query

query_router = APIRouter()

query_router.add_api_route("/", handle_query, methods=["POST"])