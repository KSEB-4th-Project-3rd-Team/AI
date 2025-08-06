using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;

public class AgentController : MonoBehaviour
{
    public int agentID;
    public AMR_Agent amrAgent;
    public Queue<Mission> missionQueue = new Queue<Mission>();
    public Mission currentMission;
    public float cellSize = 1.0f;

    // 실험명 변수 (Inspector에서 지정)
    public string expName = "RewardTestV1";

    // --- 미션 통계 저장 구조체 & 리스트 ---
    public class MissionStat
    {
        public int agentID;
        public int missionID;
        public string missionType;
        public float startTime;
        public float endTime;
        public float totalReward;
        public int collisionCount;
        public int stuckCount;
        public bool success;
    }
    public List<MissionStat> missionStats = new List<MissionStat>();

    // --- 통계 누적 변수 ---
    [Header("미션 통계(누적)")]
    public float lastMissionPathLength = 0f;
    public float lastMissionElapsedTime = 0f;
    public int lastMissionCollisionCount = 0;
    public int lastMissionStuckCount = 0;
    public bool lastMissionSuccess = false;
    public float lastMissionRewardSum = 0f;

    // --- 에피소드별 Raw 로그 (Step별 기록) ---
    private StringBuilder rawLogSB = new StringBuilder();
    private int currentStepCount = 0;
    private float missionStartTime = 0f;
    private Vector3 lastStepPos;
    private float cumulativePathLength = 0f;

    void Start()
    {
        amrAgent.agentController = this; // AMR_Agent에 컨트롤러 연결 (필수)
    }

    public void AssignNextMission()
    {
        if (missionQueue.Count > 0)
        {
            currentMission = missionQueue.Dequeue();

            // 미션 시작시점 초기화
            missionStartTime = Time.time;
            currentStepCount = 0;
            cumulativePathLength = 0f;
            lastStepPos = currentMission.GetStartWorldPos(cellSize);
            InitRawLog(currentMission);

            // 목표 세팅 (중간목표 없음 가정, 필요시 확장)
            amrAgent.SetMissionTarget(
                currentMission.GetStartWorldPos(cellSize),
                currentMission.GetGoalWorldPos(cellSize),
                currentMission.type,
                Vector3.zero, // intermediate 목표 없으면 zero
                false         // hasIntermediate = false
            );

            // 미션 상태 전환
            if (currentMission.type == "INBOUND" && currentMission.status == "WAIT")
                currentMission.status = "DOING";
            else if (currentMission.type == "OUTBOUND" && currentMission.status == "READY")
                currentMission.status = "DOING";
        }
        else
        {
            currentMission = null;
            Debug.Log("미션 큐 없음: Agent " + agentID);
        }
    }

    // ========== Raw 로그 Step별 추가 ==========
    public void LogStep(int action, float reward, Vector3 position)
    {
        float stepPath = Vector3.Distance(position, lastStepPos);
        cumulativePathLength += stepPath;
        lastStepPos = position;
        currentStepCount++;
        // Format: step, time, action, reward, pos.x, pos.y, pos.z, 누적경로
        rawLogSB.AppendLine($"{currentStepCount},{Time.time - missionStartTime},{action},{reward},{position.x:F2},{position.y:F2},{position.z:F2},{cumulativePathLength:F2}");
    }

    private void InitRawLog(Mission mission)
    {
        rawLogSB.Clear();
        rawLogSB.AppendLine("step,time,action,reward,pos_x,pos_y,pos_z,path_length");
        cumulativePathLength = 0f;
    }

    private void SaveRawLogToFile()
    {
        if (currentMission == null) return;
        string path = MakeLogFileName($"RawLog_Agent{agentID}_Mission{currentMission.order_item_id}", expName);
        File.WriteAllText(path, rawLogSB.ToString());
        Debug.Log($"에피소드 Raw 로그 저장: {path}");
    }

    // 파일명 자동 생성 함수
    private string MakeLogFileName(string logType, string expName)
    {
        string folder = Path.Combine(Application.dataPath, "ExperimentLogs");
        if (!Directory.Exists(folder)) Directory.CreateDirectory(folder);
        string name = logType;
        if (!string.IsNullOrEmpty(expName))
            name += $"_{expName}";
        name += $"_{System.DateTime.Now:yyyyMMdd_HHmmss}";
        return Path.Combine(folder, name + ".csv");
    }

    // ========== 미션 종료 시점 (AMR_Agent에서 호출) ==========
    public void OnMissionCompleted(float rewardSum, int collisionCount, int stuckCount, bool isSuccess)
    {
        if (currentMission != null)
        {
            lastMissionElapsedTime = Time.time - missionStartTime;
            lastMissionPathLength = cumulativePathLength;
            lastMissionCollisionCount = collisionCount;
            lastMissionStuckCount = stuckCount;
            lastMissionSuccess = isSuccess;
            lastMissionRewardSum = rewardSum;
            SaveRawLogToFile();

            // 통계 누적 (모든 값 저장)
            missionStats.Add(new MissionStat {
                agentID = agentID,
                missionID = currentMission.order_item_id,
                missionType = currentMission.type,
                startTime = missionStartTime,
                endTime = Time.time,
                totalReward = rewardSum,
                collisionCount = collisionCount,
                stuckCount = stuckCount,
                success = isSuccess
            });

            currentMission.status = "DONE";
            MissionManager.Instance.OnMissionDone(this, currentMission);
        }
        AssignNextMission(); // 다음 미션 즉시 할당
    }

    // 실패도 별도 호출
    public void OnMissionFailed(float rewardSum, int collisionCount, int stuckCount)
    {
        if (currentMission != null)
        {
            lastMissionElapsedTime = Time.time - missionStartTime;
            lastMissionPathLength = cumulativePathLength;
            lastMissionCollisionCount = collisionCount;
            lastMissionStuckCount = stuckCount;
            lastMissionSuccess = false;
            lastMissionRewardSum = rewardSum;
            SaveRawLogToFile();

            missionStats.Add(new MissionStat
            {
                agentID = agentID,
                missionID = currentMission.order_item_id,
                missionType = currentMission.type,
                startTime = missionStartTime,
                endTime = Time.time,
                totalReward = rewardSum,
                collisionCount = collisionCount,
                stuckCount = stuckCount,
                success = false
            });

            currentMission.status = "FAILED";
            MissionManager.Instance.OnMissionFailed(this, currentMission);
        }
        AssignNextMission(); // 다음 미션 즉시 할당
    }
}
