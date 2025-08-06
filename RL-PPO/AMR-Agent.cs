using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.IO;

public class AMR_Agent : Agent
{
    [Header("Movement")]
    public float moveSpeed = 5f;
    public float rotateSpeed = 100f;
    private Rigidbody rBody;

    [Header("Environment")]
    public WarehouseEnvironment warehouseEnv; // Inspector에서 직접 할당
    public AgentController agentController;

    // 목표 관련 변수
    private Vector3 currentTargetPosition;    // 최종 목표 위치
    private Vector3 startTargetPosition;      // 미션 시작 좌표
    private Vector3 intermediateTargetPosition;
    private bool hasIntermediateTarget = false;
    private int targetPhase = 0;              // 0: 중간목표, 1: 최종목표
    private bool isInboundScenario;

    [Header("Sensors")]
    public float rayDistance = 10f;
    public float[] rayAngles = { -90, -45, 0, 45, 90 };

    [Header("Rewards")]
    private float previousDistanceToTarget;
    private Vector3 previousPosition;
    private float stuckTimer = 0f;
    private int previousLane = -1;
    private bool isResetting = false;

    // 회피 관련
    private bool isAvoidingAMR = false;
    private float avoidanceTimer = 0f;

    // 통계 변수
    private float totalReward = 0f;
    private int collisionCount = 0;
    private int stuckCount = 0;
    private int lastAction = 0;

    // ========== 목표 세팅 ==========
    // AgentController에서 호출!
    public void SetMissionTarget(
        Vector3 start,
        Vector3 goal,
        string missionType,
        Vector3 intermediate,
        bool hasIntermediate
    )
    {
        // 위치 및 상태 세팅
        startTargetPosition = start;
        currentTargetPosition = goal;
        isInboundScenario = (missionType == "INBOUND");
        intermediateTargetPosition = intermediate;
        hasIntermediateTarget = hasIntermediate;
        targetPhase = 0;

        // 위치 초기화
        transform.position = startTargetPosition;
        transform.rotation = Quaternion.identity;

        // 거리 보상 초기화
        previousDistanceToTarget = Vector3.Distance(transform.position, GetActiveTargetPosition());
        previousPosition = transform.position;

        stuckTimer = 0f;
        previousLane = GetCurrentLane();
        isResetting = false;

        // 통계 변수도 초기화
        totalReward = 0f;
        collisionCount = 0;
        stuckCount = 0;
        lastAction = 0;

        Debug.Log($"{name} - 미션 할당 (Start:{startTargetPosition}, Goal:{currentTargetPosition}, Type:{missionType})");
    }

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
        if (warehouseEnv == null)
            Debug.LogError("WarehouseEnvironment가 할당되지 않았습니다!");
    }

    public override void OnEpisodeBegin()
    {
        isResetting = true;
        rBody.linearVelocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        isResetting = false;
        // 통계 초기화는 SetMissionTarget에서 진행
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. 자신의 정보 (6개)
        sensor.AddObservation((transform.position.x - warehouseEnv.mapWidth / 2f) / (warehouseEnv.mapWidth / 2f));
        sensor.AddObservation((transform.position.z - warehouseEnv.mapHeight / 2f) / (warehouseEnv.mapHeight / 2f));
        sensor.AddObservation(rBody.linearVelocity.x / moveSpeed);
        sensor.AddObservation(rBody.linearVelocity.z / moveSpeed);
        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);

        // 2. 현재 목표 정보 (3개)
        Vector3 activeTarget = GetActiveTargetPosition();
        Vector3 dirToTarget = (activeTarget - transform.position).normalized;
        sensor.AddObservation(dirToTarget.x);
        sensor.AddObservation(dirToTarget.z);
        sensor.AddObservation(Vector3.Distance(transform.position, activeTarget) / warehouseEnv.maxDistance);

        // 3. 시나리오 정보 (3개)
        sensor.AddObservation(isInboundScenario ? 1f : 0f);
        sensor.AddObservation(targetPhase);                    // 현재 단계
        sensor.AddObservation(hasIntermediateTarget ? 1f : 0f);

        // 4. 통로 정보 (2개 추가)
        int currentLane = GetCurrentLane();
        int targetLane = GetTargetLane();
        sensor.AddObservation(currentLane / 11f);
        sensor.AddObservation(currentLane == targetLane ? 1f : 0f);

        // 5. 레이캐스트 센서
        foreach (float angle in rayAngles)
        {
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, rayDistance))
            {
                sensor.AddObservation(1f);
                sensor.AddObservation(hit.distance / rayDistance);
                float tagValue = 0f;
                switch (hit.collider.tag)
                {
                    case "Wall": tagValue = 0.2f; break;
                    case "Rack": tagValue = 0.4f; break;
                    case "Table": tagValue = 0.6f; break;
                    case "AMR": tagValue = 0.8f; break;
                    case "TempObstacle": tagValue = 1.0f; break;
                }
                sensor.AddObservation(tagValue);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(1f);
                sensor.AddObservation(0f);
            }
        }

        // 6. 다른 AMR 정보 (난이도 높을 때)
        if (warehouseEnv.difficultyParam > 0.7f)
        {
            List<AMR_Agent> nearbyAMRs = warehouseEnv.GetNearbyAMRs(transform.position, 10f, this);
            for (int i = 0; i < 2; i++)
            {
                if (i < nearbyAMRs.Count)
                {
                    Vector3 relativePos = nearbyAMRs[i].transform.position - transform.position;
                    sensor.AddObservation(relativePos.x / 10f);
                    sensor.AddObservation(relativePos.z / 10f);
                    sensor.AddObservation(nearbyAMRs[i].rBody.linearVelocity.x / moveSpeed);
                    sensor.AddObservation(nearbyAMRs[i].rBody.linearVelocity.z / moveSpeed);
                }
                else
                {
                    sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f);
                }
            }
        }
    }

    private Vector3 GetActiveTargetPosition()
    {
        if (hasIntermediateTarget && targetPhase == 0)
            return intermediateTargetPosition;
        return currentTargetPosition;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int moveAction = actions.DiscreteActions[0];
        lastAction = moveAction;

        // 움직임 처리
        switch (moveAction)
        {
            case 0: break; // 정지
            case 1: transform.Rotate(0, -rotateSpeed * Time.deltaTime, 0); break;
            case 2: rBody.MovePosition(transform.position + transform.forward * moveSpeed * Time.deltaTime); break;
            case 3: transform.Rotate(0, rotateSpeed * Time.deltaTime, 0); break;
            case 4: rBody.MovePosition(transform.position - transform.forward * moveSpeed * 0.7f * Time.deltaTime); break;
        }

        CalculateRewards();
        totalReward += GetCumulativeReward();

        // === Raw Log 기록 (AgentController가 LogStep 호출) ===
        if (agentController != null)
            agentController.LogStep(lastAction, GetCumulativeReward(), transform.position);
    }

    private void CalculateRewards()
    {
        // 시간 패널티
        AddReward(-0.0003f);

        // 목표 접근 보상
        float currentDistance = Vector3.Distance(transform.position, GetActiveTargetPosition());
        float deltaDistance = previousDistanceToTarget - currentDistance;
        if (deltaDistance > 0) AddReward(0.03f * deltaDistance);
        else AddReward(-0.003f * Mathf.Abs(deltaDistance));
        previousDistanceToTarget = currentDistance;

        // 목표 도달 체크
        if (currentDistance < 2f)
        {
            // 중간 목표 도달
            if (hasIntermediateTarget && targetPhase == 0)
            {
                targetPhase = 1;
                AddReward(0.5f);
                previousDistanceToTarget = Vector3.Distance(transform.position, currentTargetPosition);
            }
            // 최종 목표 도달
            else
            {
                AddReward(1.5f);
                float timeBonus = Mathf.Max(0, (MaxStep - StepCount) / (float)MaxStep) * 0.5f;
                AddReward(timeBonus);
                Debug.Log($"{name} 미션 완료! Bonus:{timeBonus:F2}");
                CompleteMission(true);
            }
        }

        // 정체 상태 벌점
        float positionDelta = Vector3.Distance(transform.position, previousPosition);
        CheckAMRAvoidance();

        if (positionDelta < 0.3f)
        {
            stuckTimer += Time.deltaTime;
            if (!isAvoidingAMR)
            {
                if (stuckTimer > 1f)
                {
                    AddReward(-0.01f);
                    if (stuckTimer > 5f)
                    {
                        stuckCount++;
                        AddReward(-0.5f);
                        Debug.Log("AMR Stuck For Too Long!");
                        FailMission();
                    }
                }
            }
            else
            {
                if (stuckTimer > 2f)
                {
                    AddReward(-0.002f);
                }
            }
        }
        else
        {
            stuckTimer = 0f;
            if (positionDelta > 1.0f && !isAvoidingAMR)
                AddReward(0.0005f);
        }
        previousPosition = transform.position;

        // 잘못된 통로 진입 벌점
        int currentLane = GetCurrentLane();
        int targetLane = GetTargetLane();
        if (targetLane >= 0 && currentLane >= 0 && currentLane != targetLane)
        {
            AddReward(-0.001f);
        }
        if (!isResetting && previousLane >= 0 && currentLane == -1 && targetLane >= 0 && targetLane != previousLane)
        {
            AddReward(0.02f);
            Debug.Log($"AMR Escaped From Wrong Lane {previousLane}");
        }
        previousLane = currentLane;
    }

    private void CheckAMRAvoidance()
    {
        bool amrNearby = false;
        if (warehouseEnv.difficultyParam > 0.7f)
        {
            List<AMR_Agent> nearbyAMRs = warehouseEnv.GetNearbyAMRs(transform.position, 3f, this);
            if (nearbyAMRs.Count > 0)
            {
                amrNearby = true;
                foreach (var otherAMR in nearbyAMRs)
                {
                    Vector3 toOther = (otherAMR.transform.position - transform.position).normalized;
                    float facingDot = Vector3.Dot(transform.forward, toOther);
                    float otherFacingDot = Vector3.Dot(otherAMR.transform.forward, -toOther);
                    if (facingDot > 0.8 && otherFacingDot > 0.8f)
                    {
                        if (Random.value < 0.5f)
                        {
                            AddReward(0.01f);
                            Debug.Log($"{name} yields to avoid deadlock");
                        }
                    }
                }
            }
        }
        if (amrNearby)
        {
            isAvoidingAMR = true;
            avoidanceTimer = 0f;
        }
        else
        {
            avoidanceTimer += Time.deltaTime;
            if (avoidanceTimer > 0.5f)
                isAvoidingAMR = false;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall") ||
            collision.gameObject.CompareTag("Rack") ||
            collision.gameObject.CompareTag("Table") ||
            collision.gameObject.CompareTag("Reprocessing"))
        {
            collisionCount++;
            AddReward(-1.0f);
            FailMission();
        }
        else if (collision.gameObject.CompareTag("AMR"))
        {
            collisionCount++;
            if (warehouseEnv.difficultyParam < 0.85f)
            {
                AddReward(-0.5f);
            }
            else
            {
                AddReward(-0.8f);
                FailMission();
                AMR_Agent otherAgent = collision.gameObject.GetComponent<AMR_Agent>();
                if (otherAgent != null)
                {
                    otherAgent.AddReward(-0.8f);
                    otherAgent.FailMission();
                }
            }
        }
        else if (collision.gameObject.CompareTag("TempObstacle"))
        {
            collisionCount++;
            AddReward(-0.5f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;
        if (Input.GetKey(KeyCode.A)) discreteActionsOut[0] = 1;
        else if (Input.GetKey(KeyCode.D)) discreteActionsOut[0] = 3;
        else if (Input.GetKey(KeyCode.S)) discreteActionsOut[0] = 4;
        else if (Input.GetKey(KeyCode.W)) discreteActionsOut[0] = 2;
    }

    private int GetCurrentLane()
    {
        float x = transform.position.x;
        float z = transform.position.z;
        if (z < warehouseEnv.rackLineEndPosZ || z > warehouseEnv.rackLineStartPosZ)
            return -1;
        int closestLane = -1;
        float minDistance = float.MaxValue;
        for (int i = 0; i < warehouseEnv.rackLineBaseXCoordinates.Length; i++)
        {
            float distance = Mathf.Abs(x - warehouseEnv.rackLineBaseXCoordinates[i]);
            if (distance < warehouseEnv.laneWidth / 2f)
            {
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestLane = i;
                }
            }
        }
        return closestLane;
    }

    private int GetTargetLane()
    {
        Vector3 targetPos = GetActiveTargetPosition();
        if (targetPos.z < warehouseEnv.rackLineEndPosZ ||
            targetPos.z > warehouseEnv.rackLineStartPosZ)
            return -1;
        int closestLane = -1;
        float minDistance = float.MaxValue;
        for (int i = 0; i < warehouseEnv.rackLineBaseXCoordinates.Length; i++)
        {
            float distance = Mathf.Abs(targetPos.x - warehouseEnv.rackLineBaseXCoordinates[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestLane = i;
            }
        }
        return closestLane;
    }

    private void CompleteMission(bool isSuccess = true)
    {
        if (agentController != null)
            agentController.OnMissionCompleted(totalReward, collisionCount, stuckCount, isSuccess);
        EndEpisode();
    }
    private void FailMission()
    {
        if (agentController != null)
            agentController.OnMissionFailed(totalReward, collisionCount, stuckCount);
        EndEpisode();
    }

    void OnDrawGizmos()
    {
        if (Application.isPlaying && warehouseEnv != null && warehouseEnv.activeAMRCount <= 1)
        {
            Gizmos.color = Color.black;
            Gizmos.DrawWireSphere(GetActiveTargetPosition(), 2f);
            if (hasIntermediateTarget)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawWireSphere(intermediateTargetPosition, 1.5f);
                Gizmos.color = Color.red;
                Gizmos.DrawWireSphere(currentTargetPosition, 1.5f);
                Gizmos.color = Color.blue;
                Gizmos.DrawLine(transform.position, GetActiveTargetPosition());
            }
        }
    }
}
