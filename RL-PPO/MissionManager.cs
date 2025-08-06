using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using System.Text;

public class MissionManager : MonoBehaviour
{
    public static MissionManager Instance; // 싱글톤 패턴
    public MissionLoader loader;
    public List<AgentController> agents;
    public List<Mission> missionList = new List<Mission>();

    // 통계 기록 구조체
    [System.Serializable]
    public class StatRecord
    {
        public int episode;
        public int agentID;
        public int missionID;
        public string missionType;
        public float pathLength;
        public float elapsedTime;
        public int collisionCount;
        public int stuckCount;
        public bool success;
        public float totalReward;
    }
    private List<StatRecord> statRecords = new List<StatRecord>();
    private int episodeCounter = 0;

    void Awake()
    {
        Instance = this;
    }

    void Start()
    {
        episodeCounter = 0;
        loader.DistributeMissionsToAgents(agents);

        foreach (var agent in agents)
            agent.AssignNextMission();
    }

    // 미션 완료/상태전환
    public void OnMissionDone(AgentController agent, Mission completedMission)
    {
        completedMission.status = "DONE";
        Debug.Log($"Agent {agent.agentID} Mission {completedMission.order_item_id}({completedMission.type}) 완료");

        // INBOUND 완료 → OUTBOUND READY 전환
        if (completedMission.type == "INBOUND")
        {
            foreach (var m in missionList)
            {
                if (m.type == "OUTBOUND" && m.rack_id == completedMission.rack_id && m.status == "PENDING")
                {
                    m.status = "READY";
                    Debug.Log($"OUTBOUND 미션 READY 상태로 전환: {m.rack_id} ({m.order_item_id})");
                }
            }
        }

        // 통계 기록
        StatRecord stat = new StatRecord
        {
            episode = episodeCounter,
            agentID = agent.agentID,
            missionID = completedMission.order_item_id,
            missionType = completedMission.type,
            pathLength = agent.lastMissionPathLength,
            elapsedTime = agent.lastMissionElapsedTime,
            collisionCount = agent.lastMissionCollisionCount,
            stuckCount = agent.lastMissionStuckCount,
            success = agent.lastMissionSuccess,
            totalReward = agent.lastMissionRewardSum
        };
        statRecords.Add(stat);

        // OUTBOUND READY로 바뀐 미션 즉시 할당
        foreach (var m in missionList)
        {
            if (m.type == "OUTBOUND" && m.status == "READY" && m.assigned_agent == agent.agentID && !agent.missionQueue.Contains(m))
            {
                agent.missionQueue.Enqueue(m);
            }
        }

        agent.AssignNextMission();
    }

    public void OnMissionFailed(AgentController agent, Mission failedMission)
    {
        failedMission.status = "FAILED";
        Debug.LogWarning($"Agent {agent.agentID} Mission {failedMission.order_item_id} FAILED!");

        StatRecord stat = new StatRecord
        {
            episode = episodeCounter,
            agentID = agent.agentID,
            missionID = failedMission.order_item_id,
            missionType = failedMission.type,
            pathLength = agent.lastMissionPathLength,
            elapsedTime = agent.lastMissionElapsedTime,
            collisionCount = agent.lastMissionCollisionCount,
            stuckCount = agent.lastMissionStuckCount,
            success = false,
            totalReward = agent.lastMissionRewardSum
        };
        statRecords.Add(stat);

        agent.AssignNextMission();
    }

    public void SaveStatisticsToCSV(string fileName = "SimulationStats.csv")
    {
        string filePath = Path.Combine(Application.dataPath, fileName);
        StringBuilder sb = new StringBuilder();

        sb.AppendLine("episode,agentID,missionID,missionType,pathLength,elapsedTime,collisionCount,stuckCount,success,totalReward");
        foreach (var r in statRecords)
        {
            sb.AppendLine($"{r.episode},{r.agentID},{r.missionID},{r.missionType},{r.pathLength},{r.elapsedTime},{r.collisionCount},{r.stuckCount},{r.success},{r.totalReward}");
        }
        File.WriteAllText(filePath, sb.ToString());
        Debug.Log($"[통계 기록] {filePath}");
    }

    public void PrintStatisticsSummary()
    {
        if (statRecords.Count == 0) return;

        float avgLength = 0, avgTime = 0, avgReward = 0;
        int totalCollisions = 0, totalStucks = 0, succCount = 0, failCount = 0;

        foreach (var r in statRecords)
        {
            avgLength += r.pathLength;
            avgTime += r.elapsedTime;
            avgReward += r.totalReward;
            totalCollisions += r.collisionCount;
            totalStucks += r.stuckCount;
            if (r.success) succCount++; else failCount++;
        }
        avgLength /= statRecords.Count;
        avgTime /= statRecords.Count;
        avgReward /= statRecords.Count;

        float successRate = 100f * succCount / (succCount + failCount);

        Debug.Log($"[시뮬레이션 통계 요약]");
        Debug.Log($"- 평균 경로 길이: {avgLength:F2}");
        Debug.Log($"- 평균 미션 수행 시간: {avgTime:F2}초");
        Debug.Log($"- 평균 보상: {avgReward:F3}");
        Debug.Log($"- 충돌 총합: {totalCollisions}, Deadlock/Stuck 총합: {totalStucks}");
        Debug.Log($"- 성공률: {successRate:F1}%  (성공:{succCount} / 실패:{failCount})");
    }
}
