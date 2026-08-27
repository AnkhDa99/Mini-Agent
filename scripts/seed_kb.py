"""种子数据：创建 15 条常见 IT 运维排障条目，并同步 Neo4j + 创建关联关系。"""
import json
import uuid
import sys
sys.path.insert(0, 'd:/pythontest/mini_agent')

import numpy as np
import faiss
from app.core.database import SessionLocal
from app.core.neo4j import upsert_entry_node, create_relation
from app.repository import scenario_repo as repo
from app.services.embedding_service import EmbeddingService


def make_uid(prefix):
    return f'{prefix}_{uuid.uuid4().hex[:12]}'


def build_plain(title, content_json):
    data = json.loads(content_json) if isinstance(content_json, str) else content_json
    parts = [title]
    for k, v in data.items():
        if isinstance(v, list):
            parts.append(f'{k}: {"; ".join(str(x) for x in v)}')
        elif isinstance(v, dict):
            parts.append(f'{k}: {"; ".join(f"{a}={b}" for a, b in v.items())}')
        elif isinstance(v, str):
            parts.append(f'{k}: {v}')
    return '\n'.join(parts)


def seed():
    db = SessionLocal()
    emb = EmbeddingService()

    # ═══════ Templates ═══════
    templates_data = [
        ('tmpl_redis', 'Redis 故障排查', '数据库故障', 'Redis',
         'Redis 内存、主从、集群、缓存相关故障排查', 'Redis,缓存,内存,主从,集群', '⚡'),
        ('tmpl_k8s', 'Kubernetes 故障排查', '基础设施', 'K8s',
         'Pod 异常、调度失败、资源不足等 K8s 故障', 'K8s,Kubernetes,Pod,容器', '☸️'),
        ('tmpl_nginx', 'Nginx 故障排查', '中间件', 'Nginx',
         '502/504、反向代理超时、连接数瓶颈', 'Nginx,反向代理,502,负载均衡', '🌐'),
        ('tmpl_kafka', 'Kafka 故障排查', '中间件', 'Kafka',
         '消息积压、Consumer 掉线、分区异常', 'Kafka,消息队列,积压,消费者', '📨'),
        ('tmpl_es', 'Elasticsearch 故障排查', '数据库故障', 'ES',
         '集群状态异常、写入阻塞、查询超时', 'ES,Elasticsearch,集群,搜索', '🔍'),
        ('tmpl_docker', 'Docker 故障排查', '基础设施', 'Docker',
         '容器异常退出、磁盘满、网络不通', 'Docker,容器,磁盘,网络', '🐳'),
        ('tmpl_java', 'Java 应用故障排查', '应用故障', 'Java',
         'OOM、CPU 飙高、线程池满、GC 异常', 'Java,JVM,OOM,GC,线程池', '☕'),
    ]

    for uid, name, cat, sub, desc, tags, icon in templates_data:
        existing = repo.get_template_by_uid(db, uid)
        if not existing:
            repo.create_template(db, template_uid=uid, owner_id=4,
                                 name=name, category=cat, sub_category=sub,
                                 description=desc, tags=tags, icon=icon, priority=1)
            print(f'Template created: {name}')
        else:
            repo.update_template(db, uid, name=name, category=cat,
                                 sub_category=sub, description=desc, tags=tags, icon=icon)
            print(f'Template updated: {name}')

    # Fix MySQL template
    tmpl_sql = repo.get_template_by_uid(db, 'tmpl_c07cdd420419')
    if tmpl_sql:
        repo.update_template(db, 'tmpl_c07cdd420419',
                             name='MySQL 故障排查', category='数据库故障',
                             sub_category='MySQL',
                             description='MySQL 连接、主从、死锁、慢查询等故障的排查与修复',
                             tags='MySQL,数据库,连接,主从,性能', icon='🐬')

    db.commit()
    print('All templates ready\n')

    # ═══════ Entries ═══════
    entries_data = [
        # MySQL
        ('MySQL 主从复制延迟故障排查', 'tmpl_c07cdd420419', {
            'symptoms': ['从库数据滞后', 'SHOW SLAVE STATUS 显示 Seconds_Behind_Master > 0', '读写分离场景读到旧数据'],
            'environment': {'os': 'Ubuntu 22.04', 'version': 'MySQL 8.0.35'},
            'root_cause': '主库写入压力大导致 binlog 产生速度超过从库 relay log 消费速度，或从库 SQL 线程遇到锁等待',
            'solution_steps': [
                '1. SHOW SLAVE STATUS\\G 检查 Seconds_Behind_Master 和 Slave_SQL_Running',
                '2. 检查从库是否有长事务阻塞: SELECT * FROM information_schema.innodb_trx',
                '3. 开启并行复制: SET GLOBAL slave_parallel_workers=4',
                '4. 如果差距过大，考虑重建从库: mysqldump + xtrabackup',
            ],
            'prevention': ['开启并行复制 slave_parallel_workers', '主库拆分大事务', '从库使用 SSD 提升写入速度', '监控延迟 >30s 告警'],
            'severity': 'P2', 'estimated_fix_time': '15-30分钟',
        }, 'MySQL,主从,延迟,复制'),

        ('MySQL 死锁故障排查', 'tmpl_c07cdd420419', {
            'symptoms': ['事务报错 Deadlock found', '业务操作返回失败需要重试', 'error log 出现 deadlock 记录'],
            'environment': {'os': 'Ubuntu 22.04', 'version': 'MySQL 8.0.35'},
            'root_cause': '两个或多个事务互相等待对方持有的锁，形成循环依赖',
            'solution_steps': [
                '1. SHOW ENGINE INNODB STATUS\\G 查看 LATEST DETECTED DEADLOCK 段落',
                '2. 分析死锁日志中两个事务的锁请求顺序',
                '3. 调整事务中 SQL 执行顺序，保证所有事务以相同顺序访问资源',
                '4. 为相关表添加索引，避免 gap lock 范围过大导致锁冲突',
            ],
            'prevention': ['保持事务短小精悍', '统一事务中表的访问顺序', '使用乐观锁代替悲观锁', '对高频并发表添加合适索引'],
            'severity': 'P2', 'estimated_fix_time': '10-20分钟',
        }, 'MySQL,死锁,事务,锁'),

        # Redis
        ('Redis 内存满 OOM 故障排查', 'tmpl_redis', {
            'symptoms': ['Redis 响应变慢或拒绝写入', 'OOM command not allowed when used memory', 'used_memory 接近 maxmemory'],
            'environment': {'version': 'Redis 7.0', 'maxmemory': '8GB'},
            'root_cause': '写入数据量超过 maxmemory 限制且淘汰策略为 noeviction',
            'solution_steps': [
                '1. INFO memory 确认 used_memory_rss 和 maxmemory_human',
                '2. CONFIG SET maxmemory-policy allkeys-lru 开启 LRU 淘汰',
                '3. redis-cli --bigkeys 排查大 key 并优化数据结构',
                '4. 如业务允许设置过期时间 EXPIRE，必要时扩容内存',
            ],
            'prevention': ['设置 maxmemory-policy 为 allkeys-lru（避免 noeviction）', '监控内存使用率 >80% 告警', '定期清理过期 key', '使用 hash/list 压缩小对象'],
            'severity': 'P2', 'estimated_fix_time': '10分钟',
        }, 'Redis,内存,OOM,LRU,淘汰策略'),

        ('Redis 缓存雪崩故障排查', 'tmpl_redis', {
            'symptoms': ['大量请求直接打到数据库', '数据库连接数突然飙升', '缓存命中率突然下降'],
            'environment': {'version': 'Redis 7.0'},
            'root_cause': '大量缓存 key 在同一时间段集中过期，导致所有请求穿透缓存直接访问数据库',
            'solution_steps': [
                '1. 确认缓存 key 过期时间分布，抽样分析 TTL',
                '2. 为过期时间添加随机值: EXPIRE key 3600 + random(0, 600)',
                '3. 使用互斥锁保证只有一个线程重建缓存',
                '4. 设置永不过期的逻辑过期时间 + 后台异步更新',
            ],
            'prevention': ['过期时间添加 10-20% 随机抖动', '核心数据设置多级缓存', '限流降级保护数据库'],
            'severity': 'P1', 'estimated_fix_time': '30分钟',
        }, 'Redis,缓存雪崩,过期,穿透'),

        # K8s
        ('K8s CrashLoopBackOff 故障排查', 'tmpl_k8s', {
            'symptoms': ['Pod 状态 CrashLoopBackOff', 'kubectl describe pod 显示 Back-off restarting', '应用间歇不可用'],
            'environment': {'k8s_version': '1.28', 'runtime': 'containerd'},
            'root_cause': '容器启动后立即退出（exit code != 0），K8s 反复重启失败进入 CrashLoop 退避',
            'solution_steps': [
                '1. kubectl logs <pod> --previous 查看上一次容器的标准输出和错误',
                '2. kubectl describe pod <pod> 检查 Events 中是否有 OOMKilled',
                '3. 检查 ENTRYPOINT/CMD 是否正确，是否执行完就退出',
                '4. 检查资源限制: resources.limits 是否过小导致 OOM',
            ],
            'prevention': ['设置合理的 startupProbe 初始延迟', '容器入口进程前台运行', '配置合适的 resource limits', '添加 health check 端点'],
            'severity': 'P2', 'estimated_fix_time': '15分钟',
        }, 'K8s,CrashLoop,Pod,容器'),

        ('K8s OOMKilled 故障排查', 'tmpl_k8s', {
            'symptoms': ['Pod 被 K8s 杀死', 'kubectl describe 显示 OOMKilled', 'Exit Code 137'],
            'environment': {'k8s_version': '1.28', 'app': 'Java 17'},
            'root_cause': '容器使用内存超过 resources.limits.memory 限制，内核 OOM Killer 杀死进程',
            'solution_steps': [
                '1. kubectl top pod 查看实际内存使用',
                '2. 检查 JVM -Xmx 是否超过了容器内存限制',
                '3. 增大 resources.limits.memory 或优化应用内存使用',
                '4. 使用 -XX:MaxRAMPercentage=75.0 让 JVM 感知容器限制',
            ],
            'prevention': ['JVM 使用容器感知参数 MaxRAMPercentage', '设置 resources.requests 约为 limits 的 70%', '配置 Grafana 内存监控告警'],
            'severity': 'P2', 'estimated_fix_time': '20分钟',
        }, 'K8s,OOM,内存,Java'),

        ('K8s ImagePullBackOff 故障排查', 'tmpl_k8s', {
            'symptoms': ['Pod 状态 ImagePullBackOff 或 ErrImagePull', '新版本无法部署'],
            'environment': {'k8s_version': '1.28'},
            'root_cause': 'K8s 无法从镜像仓库拉取镜像：认证凭证过期、镜像 tag 不存在或网络不通',
            'solution_steps': [
                '1. kubectl describe pod 查看具体 pull 错误信息',
                '2. 检查 imagePullSecrets 是否正确配置',
                '3. 手动 crictl pull <image> 在节点上验证镜像可拉取',
                '4. 检查私有仓库凭证是否过期，重新生成 Secret',
            ],
            'prevention': ['使用镜像仓库凭证自动续期', '部署前在测试环境验证镜像可拉取', '合理使用 imagePullPolicy'],
            'severity': 'P3', 'estimated_fix_time': '10分钟',
        }, 'K8s,镜像,Pull,部署'),

        # Nginx
        ('Nginx 502 Bad Gateway 故障排查', 'tmpl_nginx', {
            'symptoms': ['客户端返回 502 Bad Gateway', 'error log 显示 upstream 连接失败', '后端服务实际在运行'],
            'environment': {'nginx_version': '1.24', 'upstream': 'Python FastAPI'},
            'root_cause': 'Nginx 无法连接到上游后端服务：进程挂了、端口不对、防火墙拦截或 keepalive 连接池耗尽',
            'solution_steps': [
                '1. tail -f /var/log/nginx/error.log 检查具体 upstream 错误',
                '2. curl http://127.0.0.1:<upstream_port>/health 确认后端可用',
                '3. 检查 upstream 块中的 server 地址和端口是否正确',
                '4. 增大 upstream keepalive 连接池: keepalive 32',
            ],
            'prevention': ['配置 upstream 健康检查', '设置 proxy_next_upstream error timeout', '后端服务配置优雅关闭', '监控 502 比例 >1% 告警'],
            'severity': 'P2', 'estimated_fix_time': '10-15分钟',
        }, 'Nginx,502,上游,反向代理'),

        ('Nginx 连接数打满故障排查', 'tmpl_nginx', {
            'symptoms': ['客户端连接超时或拒绝', 'error log: worker_connections are not enough'],
            'environment': {'nginx_version': '1.24', 'worker_connections': '1024'},
            'root_cause': '并发连接数超过 worker_connections * worker_processes 上限，后端响应慢导致连接堆积',
            'solution_steps': [
                '1. curl http://127.0.0.1/nginx_status 查看 Active connections',
                '2. 增大 worker_connections 到 4096 或更高后 reload',
                '3. 检查后端响应时间，优化慢接口',
                '4. 配置 limit_conn 和 limit_req 进行限流保护',
            ],
            'prevention': ['监控 Active connections 占比', '设置合理的 keepalive_timeout', '使用 CDN 分流静态资源'],
            'severity': 'P1', 'estimated_fix_time': '10分钟',
        }, 'Nginx,连接数,worker,性能'),

        # Kafka
        ('Kafka 消息积压故障排查', 'tmpl_kafka', {
            'symptoms': ['Consumer Lag 持续增长', '消息处理延迟增大', 'kafka-consumer-groups 显示 LAG 值很大'],
            'environment': {'kafka_version': '3.5', 'partitions': '12'},
            'root_cause': '消费者处理速度跟不上生产者写入速度：消费者实例不足、处理逻辑变慢或分区数瓶颈',
            'solution_steps': [
                '1. kafka-consumer-groups --describe 确认 LAG 量和具体分区',
                '2. 增加消费者实例数（最多等于分区数）',
                '3. 检查消费者处理逻辑是否有慢查询或外部调用超时',
                '4. 如果分区数瓶颈，增加 topic 分区数（注意消息顺序性要求）',
            ],
            'prevention': ['监控 Consumer Lag >10万 告警', '消费者批量处理提升吞吐', '分区数为消费者实例数的整数倍'],
            'severity': 'P2', 'estimated_fix_time': '20分钟',
        }, 'Kafka,积压,Lag,消费者'),

        ('Kafka Consumer 掉线故障排查', 'tmpl_kafka', {
            'symptoms': ['消费者组频繁 rebalance', '部分消费者意外退出', '消息处理中断'],
            'environment': {'kafka_version': '3.5'},
            'root_cause': '消费者心跳超时或 poll 间隔超时，Coordinator 将其踢出组触发 rebalance',
            'solution_steps': [
                '1. 检查消费者日志是否有 RebalanceInProgress 或 SessionTimeout',
                '2. 增大 session.timeout.ms 和 heartbeat.interval.ms',
                '3. 增大 max.poll.interval.ms 防止处理慢导致超时',
                '4. 确保 poll 循环中没有阻塞在非 Kafka 操作上',
            ],
            'prevention': ['session.timeout.ms 设置为 30s', 'max.poll.interval.ms 设置为 5min', '消费者处理逻辑异步化'],
            'severity': 'P2', 'estimated_fix_time': '15分钟',
        }, 'Kafka,Consumer,Rebalance,会话超时'),

        # ES
        ('ES 集群变红故障排查', 'tmpl_es', {
            'symptoms': ['集群状态 Red', '搜索和写入部分失败', '_cluster/health 返回 status: red'],
            'environment': {'es_version': '8.11', 'nodes': '3'},
            'root_cause': '至少一个主分片未分配（unassigned）：节点磁盘满、网络分区或分片损坏',
            'solution_steps': [
                '1. GET _cluster/allocation/explain 查看未分配分片原因',
                '2. GET _cat/shards?v&h=index,shard,state 列出所有未分配分片',
                '3. 磁盘满: 清理旧索引或扩容磁盘',
                '4. 节点恢复: 等待节点重新加入或执行 reroute 强制分配',
            ],
            'prevention': ['设置磁盘水位线 low=85% high=90%', '开启自动分片平衡', '监控集群状态变黄即告警'],
            'severity': 'P1', 'estimated_fix_time': '30分钟',
        }, 'ES,集群,Red,分片'),

        ('ES 写入阻塞故障排查', 'tmpl_es', {
            'symptoms': ['写入请求返回 429 Too Many Requests', '写入延迟暴增', '磁盘使用率超过水位线'],
            'environment': {'es_version': '8.11'},
            'root_cause': '磁盘使用率超过 flood_stage（95%），ES 自动将索引设为 read_only_allow_delete',
            'solution_steps': [
                '1. GET _cat/allocation?v 查看各节点磁盘使用率',
                '2. PUT */_settings {"index.blocks.read_only_allow_delete": null} 解除只读',
                '3. DELETE 不需要的旧索引释放空间',
                '4. 扩容节点或增加磁盘',
            ],
            'prevention': ['设置索引生命周期管理 ILM 自动清理', '磁盘使用率 >85% 告警', '定期 force merge 已完成索引'],
            'severity': 'P1', 'estimated_fix_time': '20分钟',
        }, 'ES,写入阻塞,磁盘,只读'),

        # Docker
        ('Docker 磁盘满故障排查', 'tmpl_docker', {
            'symptoms': ['docker pull/push 失败: no space left on device', '容器无法启动', '磁盘使用率飙到 100%'],
            'environment': {'os': 'Ubuntu 22.04', 'docker_version': '24.0'},
            'root_cause': 'Docker overlay2 积累了大量未使用的镜像、容器和卷',
            'solution_steps': [
                '1. docker system df 查看各类资源占用',
                '2. docker system prune -a -f 清理未使用的镜像、容器、网络',
                '3. docker volume prune -f 清理未使用的卷',
                '4. 检查并清理容器日志: du -sh /var/lib/docker/containers/*/',
            ],
            'prevention': ['daemon.json 配置日志轮转: max-size=10m max-file=3', '定期 docker system prune', '监控 /var/lib/docker 使用率'],
            'severity': 'P3', 'estimated_fix_time': '10分钟',
        }, 'Docker,磁盘,清理,overlay2'),

        # Java
        ('Java 内存泄漏故障排查', 'tmpl_java', {
            'symptoms': ['应用运行一段时间后变慢', 'Full GC 越来越频繁但回收效果差', '堆内存使用率持续增长不回落'],
            'environment': {'jdk_version': 'OpenJDK 17', 'heap_size': '4GB'},
            'root_cause': '代码中存在对象引用未释放，GC Roots 可达导致 GC 无法回收',
            'solution_steps': [
                '1. jmap -histo:live <pid> | head -50 查看内存中对象分布',
                '2. jmap -dump:format=b,file=heap.hprof <pid> 导出堆转储',
                '3. 使用 Eclipse MAT 分析 heap.hprof，查看 Dominator Tree',
                '4. 定位到持有大量对象的类后检查是否有静态集合、ThreadLocal 未清理',
            ],
            'prevention': ['Code Review 关注集合类和 ThreadLocal 清理', '集成 JFR 持续监控', '上线后观察堆内存趋势'],
            'severity': 'P2', 'estimated_fix_time': '1-2小时',
        }, 'Java,内存泄漏,JVM,GC'),

        ('Java CPU 飙高故障排查', 'tmpl_java', {
            'symptoms': ['服务器 CPU 使用率 >90%', '应用响应变慢或超时', '可能伴随 GC 频繁'],
            'environment': {'jdk_version': 'OpenJDK 17', 'cpu_cores': '8'},
            'root_cause': '某线程陷入死循环、大量正则匹配、复杂计算或频繁 Full GC',
            'solution_steps': [
                '1. top -H -p <pid> 找到 CPU 最高的线程 tid',
                '2. 将 tid 转十六进制，jstack <pid> | grep -A20 <hex_tid> 定位代码行',
                '3. 如果是 GC 线程: jstat -gcutil <pid> 1000 观察 GC 频率',
                '4. 如果是业务线程: arthas thread -n 3 直接看 CPU 最高的线程栈',
            ],
            'prevention': ['CPU 使用率 >80% 持续 5 分钟告警', '使用 arthas 定期 profiler', '上线前压测确定 CPU 基线'],
            'severity': 'P2', 'estimated_fix_time': '20-30分钟',
        }, 'Java,CPU,线程,jstack'),
    ]

    from app.services.scenario_matcher import get_scenario_matcher
    matcher = get_scenario_matcher()

    created = 0
    synced_neo4j = 0
    for title, tmpl_uid, content, tags in entries_data:
        existing = repo.list_entries(db, template_uid=tmpl_uid, status=None,
                                     offset=0, limit=100)
        existing_entry = next((e for e in existing if e.title == title), None)

        if existing_entry:
            # 已有条目：确保 Neo4j 节点同步（之前可能缺失）
            ok = upsert_entry_node(
                entry_uid=existing_entry.entry_uid,
                title=existing_entry.title,
                template_uid=existing_entry.template_uid,
            )
            if ok:
                synced_neo4j += 1
            else:
                print(f'  Neo4j sync failed for {title} (connection error)')
            continue

        content_json = json.dumps(content, ensure_ascii=False)
        entry_uid = make_uid('entry')
        plain = build_plain(title, content_json)

        entry = repo.create_entry(
            db, entry_uid=entry_uid, owner_id=4,
            template_uid=tmpl_uid, title=title,
            content_json=content_json, plain_text=plain,
            tags=tags, keywords=tags, source_type='manual',
            status='approved',
        )

        # 同步到 Neo4j 图数据库
        upsert_entry_node(
            entry_uid=entry.entry_uid,
            title=entry.title,
            template_uid=entry.template_uid,
        )

        # Index to FAISS
        try:
            vecs = emb.embed_texts([plain])
            if vecs:
                vec = np.array([vecs[0]], dtype=np.float32)
                faiss.normalize_L2(vec)
                new_id = len(matcher.id_to_entry)
                matcher.index.add_with_ids(vec, np.array([new_id], dtype=np.int64))
                matcher.id_to_entry.append(entry_uid)
                matcher.entry_to_id[entry_uid] = new_id
        except Exception as e:
            print(f'  FAISS index failed for {title}: {e}')

        created += 1
        print(f'  Created: {title}')

    if synced_neo4j > 0:
        print(f'\nNeo4j nodes synced for {synced_neo4j} existing entries')

    db.commit()
    matcher._save()
    print(f'\nTotal created: {created}')
    print(f'FAISS total: {matcher.index.ntotal}')

    # ═══════ Neo4j 知识图谱关系 ═══════
    # 收集所有已创建条目的 entry_uid（按 title 匹配）
    all_entries = repo.list_entries(db, status=None, offset=0, limit=500)
    title_to_uid = {e.title: e.entry_uid for e in all_entries}
    print(f'\nDB entries found: {len(title_to_uid)}')

    relations = [
        # MySQL 内部关联
        ('MySQL 主从复制延迟故障排查', 'MySQL 死锁故障排查', 'related_to', 0.7),
        # Redis 内部关联
        ('Redis 内存满 OOM 故障排查', 'Redis 缓存雪崩故障排查', 'related_to', 0.8),
        # K8s 内部关联: CrashLoop → OOMKilled → ImagePullBackOff
        ('K8s CrashLoopBackOff 故障排查', 'K8s OOMKilled 故障排查', 'caused_by', 0.9),
        ('K8s OOMKilled 故障排查', 'K8s ImagePullBackOff 故障排查', 'related_to', 0.5),
        # Nginx 内部关联
        ('Nginx 502 Bad Gateway 故障排查', 'Nginx 连接数打满故障排查', 'related_to', 0.7),
        # Kafka 内部关联: Consumer掉线 → 消息积压
        ('Kafka Consumer 掉线故障排查', 'Kafka 消息积压故障排查', 'caused_by', 0.85),
        # ES 内部关联: 磁盘满 → 写入阻塞 → 集群变红
        ('ES 写入阻塞故障排查', 'ES 集群变红故障排查', 'caused_by', 0.9),
        # Java 内部关联
        ('Java 内存泄漏故障排查', 'Java CPU 飙高故障排查', 'related_to', 0.7),
        # 跨模版关联
        ('MySQL 死锁故障排查', 'Java 内存泄漏故障排查', 'similar_to', 0.5),
        ('K8s OOMKilled 故障排查', 'Docker 磁盘满故障排查', 'related_to', 0.4),
        ('Redis 缓存雪崩故障排查', 'Nginx 连接数打满故障排查', 'caused_by', 0.6),
        ('K8s CrashLoopBackOff 故障排查', 'Docker 磁盘满故障排查', 'related_to', 0.4),
        ('Java CPU 飙高故障排查', 'K8s OOMKilled 故障排查', 'related_to', 0.6),
    ]

    relation_count = 0
    missing_titles = set()
    for from_title, to_title, rel_type, weight in relations:
        from_uid = title_to_uid.get(from_title)
        to_uid = title_to_uid.get(to_title)
        if not from_uid:
            missing_titles.add(from_title)
        if not to_uid:
            missing_titles.add(to_title)
        if from_uid and to_uid:
            ok = create_relation(from_uid, to_uid, relation_type=rel_type, weight=weight)
            if ok:
                relation_count += 1
                print(f'  Relation [{relation_count}]: {from_title[:30]}... -[{rel_type}]-> {to_title[:30]}...')
            else:
                print(f'  Relation FAILED (Neo4j error or nodes missing): {from_title[:30]}... -> {to_title[:30]}...')
        else:
            print(f'  SKIP (missing uid): {from_title[:30]}... -> {to_title[:30]}...')

    if missing_titles:
        print(f'\nWARNING: {len(missing_titles)} titles not found in DB:')
        for t in missing_titles:
            print(f'  - {t}')

    print(f'\nNeo4j relations created: {relation_count}')

    # ── 验证 Neo4j 状态 ──
    from app.core.neo4j import _get_driver
    try:
        driver = _get_driver()
        with driver.session(database="neo4j") as session:
            node_count = session.run(
                "MATCH (e:KnowledgeEntry) RETURN count(e) AS cnt"
            ).single()["cnt"]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            ).single()["cnt"]
            print(f'\nNeo4j Verification: {node_count} nodes, {rel_count} relationships')
            if rel_count == 0 and relation_count > 0:
                print('WARNING: Relationships were created via API but not visible in Neo4j!')
                print('Check Neo4j connection, database name, or transaction commit.')
    except Exception as exc:
        print(f'Neo4j verification failed: {exc}')

    # Also verify
    result = matcher.match(db=db, query='MySQL连接超时怎么办', top_k=5, threshold=0.5)
    print(f'Match test: {result["match_count"]} results')
    db.close()


if __name__ == '__main__':
    seed()
