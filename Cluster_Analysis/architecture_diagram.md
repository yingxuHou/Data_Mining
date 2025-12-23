# 学生宿舍监控管理系统架构图

## 现代化设计版本

```mermaid
graph TB
    %% 用户层
    subgraph Users["👥 用户层"]
        Student["🎓 学生"]
        DormAdmin["👨‍💼 宿舍管理员"]
    end

    %% 功能层
    subgraph Functions["⚡ 功能层"]
        Login["🔐 登录系统"]
        StudentMgmt["📚 学生管理"]
            StudentMgmt_CheckIn["✅ 打卡功能"]
        DormMgmt["🏠 宿管管理"]
            DormMgmt_Scoring["🌟 卫生评分"]
        AccessControl["🚪 门禁管理"]
            AccessControl_Visitor["📋 访客登记"]
    end

    %% 技术层
    subgraph Tech["💻 技术层"]
        subgraph Frontend["🌐 前端层"]
            Gateway["Spring Cloud Gateway"]
        end

        subgraph Middleware["⚙️ 中间件层"]
            Nacos["📊 Nacos"]
            Docker["🐳 Docker"]
            Redis["⚡ Redis"]
        end

        subgraph Backend["🔧 后端层"]
            SpringCloud["☁️ Spring Cloud"]
            SpringBoot["🚀 Spring Boot"]
            MySQL["🗄️ MySQL"]
        end
    end

    %% 连接关系
    Student --> Login
    DormAdmin --> Login
    Login --> StudentMgmt
    Login --> DormMgmt
    Login --> AccessControl

    StudentMgmt --> StudentMgmt_CheckIn
    DormMgmt --> DormMgmt_Scoring
    AccessControl --> AccessControl_Visitor

    StudentMgmt --> Gateway
    DormMgmt --> Gateway
    AccessControl --> Gateway

    Gateway --> Nacos
    Gateway --> Redis
    Gateway --> SpringCloud

    SpringCloud --> SpringBoot
    SpringBoot --> MySQL

    %% 样式定义
    classDef userStyle {
        fill:#FFE5B4
        stroke:#FF8C00
        stroke-width:3px
        color:#000
        font-weight:bold
        font-size:14px
    }

    classDef functionStyle {
        fill:#E6F3FF
        stroke:#4169E1
        stroke-width:2px
        color:#000
        font-weight:500
        font-size:12px
    }

    classDef subFunctionStyle {
        fill:#F0F8FF
        stroke:#6495ED
        stroke-width:1px
        color:#000
        font-size:11px
    }

    classDef techStyle {
        fill:#F0FFF0
        stroke:#32CD32
        stroke-width:2px
        color:#000
        font-weight:500
        font-size:12px
    }

    classDef frontendStyle {
        fill:#FFF0F5
        stroke:#FF69B4
        stroke-width:2px
        color:#000
        font-size:12px
    }

    classDef middlewareStyle {
        fill:#F5F5DC
        stroke:#DAA520
        stroke-width:2px
        color:#000
        font-size:12px
    }

    classDef backendStyle {
        fill:#F0F8FF
        stroke:#4682B4
        stroke-width:2px
        color:#000
        font-size:12px
    }

    %% 应用样式
    class Student,DormAdmin userStyle
    class Login,StudentMgmt,DormMgmt,AccessControl functionStyle
    class StudentMgmt_CheckIn,DormMgmt_Scoring,AccessControl_Visitor subFunctionStyle
    class Users,Functions,Tech techStyle
    class Gateway frontendStyle
    class Nacos,Docker,Redis middlewareStyle
    class SpringCloud,SpringBoot,MySQL backendStyle
```

## 设计改进说明

### 🎨 视觉优化
1. **现代化配色方案**
   - 用户层：暖橙色系 (#FFE5B4, #FF8C00)
   - 功能层：蓝色系 (#E6F3FF, #4169E1)
   - 技术层：绿色系 (#F0FFF0, #32CD32)
   - 子功能：浅蓝色渐变

2. **图标增强**
   - 每个组件添加相关emoji图标
   - 提升视觉识别度和美观度

3. **层次结构优化**
   - 清晰的三层架构划分
   - 子功能嵌套显示
   - 逻辑关系更加明确

4. **连接线优化**
   - 使用Mermaid的自动布局
   - 清晰的数据流向指示
   - 避免线条交叉混乱

### 📐 布局改进
- 采用自上而下的层次结构
- 左右对称的用户角色设计
- 中间件层居中对齐
- 统一的组件间距

### 🔧 技术实现
- 使用Mermaid语法实现
- 支持多种输出格式（PNG, SVG, PDF）
- 易于修改和维护
- 可集成到文档中

## 渲染方式
1. 在支持Mermaid的编辑器中查看（如Typora, VS Code插件）
2. 在线Mermaid编辑器：https://mermaid.live
3. GitHub/GitLab原生支持
4. 导出为图片格式