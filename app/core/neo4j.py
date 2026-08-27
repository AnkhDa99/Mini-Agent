"""Neo4j 图数据库客户端 — 管理知识条目之间的关联图谱。"""

from __future__ import annotations

import logging

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from app.core.config import settings

logger = logging.getLogger(__name__)

_driver: Driver | None = None


def _get_driver() -> Driver:
    """懒加载 Neo4j driver（单例）。"""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
        )
        logger.info("Neo4j driver initialized: %s", settings.neo4j_uri)
    return _driver


def close_neo4j():
    """关闭 Neo4j 连接。"""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


def init_neo4j_constraints():
    """初始化 Neo4j 唯一约束和索引（幂等操作）。"""
    driver = _get_driver()
    with driver.session(database="neo4j") as session:
        session.run("""
            CREATE CONSTRAINT entry_uid_unique IF NOT EXISTS
            FOR (e:KnowledgeEntry) REQUIRE e.entry_uid IS UNIQUE
        """)
        session.run("""
            CREATE INDEX entry_title_index IF NOT EXISTS
            FOR (e:KnowledgeEntry) ON (e.title)
        """)


def upsert_entry_node(entry_uid: str, title: str, template_uid: str) -> bool:
    """在 Neo4j 中创建或更新知识条目节点。返回是否成功。"""
    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("""
                MERGE (e:KnowledgeEntry {entry_uid: $entry_uid})
                SET e.title = $title, e.template_uid = $template_uid, e.updated_at = datetime()
            """, entry_uid=entry_uid, title=title, template_uid=template_uid)
            summary = result.consume()
            logger.debug("upsert_entry_node %s: %s", entry_uid, summary.counters)
        return True
    except Neo4jError:
        logger.exception("Failed to upsert entry node: %s", entry_uid)
        return False


def delete_entry_node(entry_uid: str) -> None:
    """删除知识条目节点及其所有关联关系。"""
    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            session.run("""
                MATCH (e:KnowledgeEntry {entry_uid: $entry_uid})
                DETACH DELETE e
            """, entry_uid=entry_uid)
    except Neo4jError:
        logger.exception("Failed to delete entry node: %s", entry_uid)


def create_relation(
    entry_uid_from: str,
    entry_uid_to: str,
    relation_type: str = "related_to",
    weight: float = 1.0,
) -> bool:
    """创建条目之间的关联关系。返回是否成功创建。"""
    # 允许的关系类型白名单
    allowed_types = {"related_to", "caused_by", "prerequisite_of", "similar_to", "accompanied_by"}
    if relation_type not in allowed_types:
        raise ValueError(f"Invalid relation_type: {relation_type}")

    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(f"""
                MATCH (a:KnowledgeEntry {{entry_uid: $from}})
                MATCH (b:KnowledgeEntry {{entry_uid: $to}})
                MERGE (a)-[r:{relation_type.upper()}]->(b)
                SET r.weight = $weight, r.created_at = datetime()
                RETURN count(r) AS created
            """, {"from": entry_uid_from, "to": entry_uid_to, "weight": weight})
            record = result.single()
            if record and record["created"] == 0:
                logger.warning(
                    "create_relation matched 0 nodes: %s -[%s]-> %s (nodes may not exist in Neo4j)",
                    entry_uid_from, relation_type, entry_uid_to,
                )
                return False
            return True
    except Neo4jError:
        logger.exception(
            "Failed to create relation: %s -[%s]-> %s",
            entry_uid_from, relation_type, entry_uid_to,
        )
        return False


def delete_relation(entry_uid_from: str, entry_uid_to: str, relation_type: str | None = None) -> None:
    """删除关联关系。relation_type 为 None 时删除所有类型的关系。"""
    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            if relation_type:
                session.run(f"""
                    MATCH (a:KnowledgeEntry {{entry_uid: $from}})
                    -[r:{relation_type.upper()}]->
                    (b:KnowledgeEntry {{entry_uid: $to}})
                    DELETE r
                """, {"from": entry_uid_from, "to": entry_uid_to})
            else:
                session.run("""
                    MATCH (a:KnowledgeEntry {entry_uid: $from})
                    -[r]->
                    (b:KnowledgeEntry {entry_uid: $to})
                    DELETE r
                """, {"from": entry_uid_from, "to": entry_uid_to})
    except Neo4jError:
        logger.exception("Failed to delete relation: %s -> %s", entry_uid_from, entry_uid_to)


def get_entry_relations(entry_uid: str, depth: int = 1) -> list[dict]:
    """获取知识条目的关联图谱（默认 1 跳邻居）。"""
    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(f"""
                MATCH (a:KnowledgeEntry {{entry_uid: $entry_uid}})
                -[r *1..{depth}]->
                (b:KnowledgeEntry)
                RETURN DISTINCT a.entry_uid AS from_uid, b.entry_uid AS to_uid,
                       b.title AS to_title, type(r[-1]) AS rel_type,
                       r[-1].weight AS weight
                ORDER BY weight DESC
                LIMIT 50
            """, {"entry_uid": entry_uid, "depth": depth})
            return [
                {
                    "from_uid": record["from_uid"],
                    "to_uid": record["to_uid"],
                    "to_title": record["to_title"],
                    "rel_type": record["rel_type"].lower(),
                    "weight": record["weight"],
                }
                for record in result
            ]
    except Neo4jError:
        logger.exception("Failed to get relations for: %s", entry_uid)
        return []


def get_expanded_match(
    entry_uids: list[str],
    max_related: int = 10,
) -> list[str]:
    """图谱扩展匹配：给定一批匹配到的条目，通过图谱关系找到关联条目。

    用于提升召回：命中了 A 条目 → 也推荐与 A 关联的 B、C 条目。
    """
    if not entry_uids:
        return []
    driver = _get_driver()
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (a:KnowledgeEntry)-[r]->(b:KnowledgeEntry)
                WHERE a.entry_uid IN $uids AND NOT b.entry_uid IN $uids
                RETURN DISTINCT b.entry_uid AS uid, count(r) AS relation_count
                ORDER BY relation_count DESC
                LIMIT $limit
            """, {"uids": entry_uids, "limit": max_related})
            return [record["uid"] for record in result]
    except Neo4jError:
        logger.exception("Graph expansion match failed")
        return []
