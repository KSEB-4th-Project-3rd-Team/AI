using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;

public class MissionLoader : MonoBehaviour
{
    public string csvFileName = "assign_task2.csv"; // StreamingAssets 폴더 또는 Resources에 둘 것
    public List<Mission> missionList = new List<Mission>();

    // ===== [추가] 자동으로 Agent별 Mission 큐를 만들어주는 함수 =====
    public void DistributeMissionsToAgents(List<AgentController> agents)
    {
        foreach (var agent in agents)
        {
            // INBOUND: WAIT, OUTBOUND: READY만 큐에 추가
            var myMissions = missionList.Where(m =>
                ((m.type == "INBOUND" && m.status == "WAIT") ||
                 (m.type == "OUTBOUND" && m.status == "READY")) &&
                m.assigned_agent == agent.agentID
            );
            foreach (var m in myMissions)
                agent.missionQueue.Enqueue(m);
        }
    }

    void Awake()
    {
        LoadMissionsFromCSV();
    }

    // ===== [최종] CSV 로드 함수 =====
    void LoadMissionsFromCSV()
    {
        // Application.streamingAssetsPath 권장
        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);

        if (!File.Exists(filePath))
        {
            Debug.LogError("미션 파일이 존재하지 않습니다: " + filePath);
            return;
        }

        using (var reader = new StreamReader(filePath))
        {
            bool isHeader = true;
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (isHeader)
                {
                    isHeader = false;
                    continue; // 헤더 스킵
                }
                if (string.IsNullOrWhiteSpace(line)) continue;
                var fields = line.Split(',');
                missionList.Add(new Mission(fields));
            }
        }
        Debug.Log($"미션 {missionList.Count}개 로드 완료!");
    }

    // ===== [선택] 미션 검색/필터 편의 함수들 =====

    // 지정 에이전트의 다음 미션만 반환 (상태, 타입 조건)
    public Mission GetNextMissionForAgent(int agentID)
    {
        return missionList.FirstOrDefault(m =>
            ((m.type == "INBOUND" && m.status == "WAIT") ||
             (m.type == "OUTBOUND" && m.status == "READY")) &&
            m.assigned_agent == agentID
        );
    }

    // 모든 OUTBOUND PENDING 미션 중 특정 랙, 에이전트용
    public List<Mission> GetPendingOutboundMissionsForRack(string rack_id, int agentID)
    {
        return missionList.Where(m =>
            m.type == "OUTBOUND" && m.status == "PENDING" &&
            m.rack_id == rack_id && m.assigned_agent == agentID
        ).ToList();
    }
}
